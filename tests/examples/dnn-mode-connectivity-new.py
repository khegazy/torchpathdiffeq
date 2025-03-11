import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
from torchpathdiffeq import\
    steps,\
    get_parallel_RK_solver,\
    UNIFORM_METHODS,\
    VARIABLE_METHODS\


from functools import partial
import os
import argparse


CIFAR10_DIR = './data/'
WORKERS = 4
BATCH_SIZE = 128

parser = argparse.ArgumentParser(
                    prog='DNN-mode-connectivity',
                    description='Constructs constant-accuracy curves between independent DNN modes',
                    epilog='Sucks to suck')
parser.add_argument('-w1', '--w_init')
parser.add_argument('-w2', '--w_end')
parser.add_argument('-mp', '--model_path')
parser.add_argument('-tc', '--train_curve')


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_set = CIFAR10(root=CIFAR10_DIR, train=True, transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop((32, 32), 4),
                        transforms.ToTensor(), normalize]))

trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, pin_memory=True)

testloader = DataLoader(CIFAR10(root=CIFAR10_DIR, train=False, transform=
                                    transforms.Compose([
                                        transforms.ToTensor(), normalize])),
                                batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=WORKERS, pin_memory=True)


args = parser.parse_args()

if args.w_init is not None and args.w_end is None:
    if args.model_path is None:
        raise ValueError("Please specify both ends of the curve in addition to the model path")
    raise ValueError("Both ends of the curve should be specified")

if args.w_init is None and args.w_end is not None:
    if args.model_path is None:
        raise ValueError("Please specify both ends of the curve in addition to the model path")
    raise ValueError("Both ends of the curve should be specified")


### config dictionary for the tests ###
test_config = {
    'epochs': 100,
    'epochs_integrator': 100,
    'batch_size': 32,
    'criterion': {
        'CE': nn.CrossEntropyLoss()
    },
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'dataset': str(args.dataset),
    'LR': 1e-3,
    'resnet_depth': int(args.resnet_depth),
    'cifar10_path': './data',
    'checkpoints_path': './checkpoints',
    'sampling_type': steps.ADAPTIVE_UNIFORM,
    'method': 'adaptive_heun',
    'atol': 1e-5,
    'rtol': 1e-3,
    'max_batches': 256,
    'model_init_path': "",
    'model_end_path': "",
    'w1_path': args.w_init,
    'w2_path': args.w_end,
    'curve_path': "./mode_test_path_t_detached.pt"
}

test_config['optims'] = {
        'Adam': partial(optim.Adam, lr=test_config["LR"])
}


### Describing the network used for the path ###

class CurveNet(nn.Module):
    def __init__(self, nn_dims: int, w1: torch.Tensor, w2: torch.Tensor, criterion: nn.Module):#, model: nn.Module):
        super(CurveNet, self).__init__()
        self.fc1 = nn.Linear(1, 100) # take input `t`
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, nn_dims) #... and return a set of weights in the weight-space
        self.w1 = w1 # should be passed as output of a `parameters_to_vector` call
        self.w2 = w2 # Ditto
        self.criterion = criterion
        self.nn_dims = nn_dims
        # self.model = model #TODO: Keeping the initial model on hand for evaluation... or not?

    def forward(self, t): # `t\in [0, 1]`
        #TODO: enforce `t\in [0, 1]` --- this will be respected by `get_parallel_RK_solver`, confirm
        #TODO: How do I enforce CurveNet(0) = w1 && CurveNet(1) = w2?
        #TODO: How will a `CurveNet` instance actually be trained?
        coeffs = [1 - t, t, 1 - torch.cos(2 * math.pi * t)]
        # coeffs = [1 - t, t]
        t_init = torch.tensor([0]).to(test_config['device']).type(torch.float)
        t_final = torch.tensor([1]).to(test_config['device']).type(torch.float)
        f1 = F.tanh(self.fc3(F.tanh(self.fc2(F.tanh(self.fc1(t_final))))))
        f0 = F.tanh(self.fc3(F.tanh(self.fc2(F.tanh(self.fc1(t_init))))))
        t = F.tanh(self.fc1(t))
        t = F.tanh(self.fc2(t))
        t = F.tanh(self.fc3(t)) # `t` now represents a point in weight-space

        # enforcing that `Phi_t` evaluates to w_1 @ 0 and w_2 @ 1
        if len(t.shape) < 2:
            # return (coeffs[0] * self.w1 + coeffs[1] * self.w2 + coeffs[2] * t).view(1, self.nn_dims)
            return (coeffs[0] * (t - f0 + self.w1) + coeffs[1] * (t - f1 + self.w2)).view(1, self.nn_dims)
        else:
            # return coeffs[0] * self.w1 + coeffs[1] * self.w2 + coeffs[2] * t
            return coeffs[0] * (t - f0 + self.w1) + coeffs[1] * (t - f1 + self.w2)
        # (1-t) * w1 + t * (f(t) - f(1) + w2)
        # (1 - t) * (f(t) - f(0) + w1) + t * (f(t) - f(1) + w2)

class CurveNetLoss(nn.Module):
    """
    Computes the loss L(w) := L(Y, model(X; w)), multithreaded
    Should be passed to the integrator
    NOTE: Need to verify thread-safety of this implementation
    """
    def __init__(self, criterion: nn.Module, model: nn.Module, curve: CurveNet):
        super(CurveNetLoss, self).__init__()
        self.criterion = criterion
        self.model = model # we don't need the trained weights here, just require the class itself
        self.curve = curve

        self._threads = 10

        self._models: list[nn.Module] = [self.model] * self._threads

    def forward(self, t):#, X, Y):
        #TODO: pass in the weights here instead of time
        w = self.curve(t) # grab the curves output
        models_per_thread: int = w.shape[0] / self._threads

        # Mapping between models and threads
        model_to_thread = [(dim, dim % models_per_thread) for dim in range(w.shape[0])]

        loss_ls = [0] * w.shape[0] # pre-allocating the loss array
        self.model.eval()

        def worker(dim, thread_id):
            eval_running_loss = 0.0
            vector_to_parameters(w[dim, :].view(-1), self._models[thread_id].parameters())

            for i, eval_data in enumerate(testloader, 0):
                eval_X, eval_Y = eval_data
                eval_X = eval_X.to(test_config['device'])
                eval_Y = eval_Y.to(test_config['device'])
                eval_outputs = self.model(eval_X)

                loss = self.criterion(eval_outputs, eval_Y)
                eval_running_loss += loss.detach().item()
            loss_ls[dim] = eval_running_loss
            pass

        mp.set_start_method('spawn')  # Use 'spawn' method for better compatibility
        with mp.Pool(processes=self._threads) as pool:
            # Pass an index (thread_id) along with parameters
            pool.starmap(worker, [(i, w[dim, :], self._models[i]) for (dim, i) in model_to_thread])

        return torch.tensor(loss_ls).to(test_config['device']).view(len(loss_ls), 1)

        # for dim in range(w.shape[0]):
        #     vector_to_parameters(w[dim, :].view(-1), self.model.parameters())
        #     loss.append(self.criterion(self.model(X), Y).view(1,1))
        # return torch.tensor(loss).to(test_config['device']).view(len(loss), 1)

### Describing the network used for the path ###


### Curve eval on a straight line, START

def eval_line(w1: torch.Tensor, w2: torch.Tensor, plain_model: nn.Module, num_points: int):
    # Generating the linearly interpolated points
    t_values = torch.linspace(0, 1, num_points)
    points = torch.lerp(w1, w2, t_values.unsqueeze(1))
    criterion = test_config['criterion']['CE']
    loss_ls = []

    print("Line\n")
    for i in range(num_points):
        eval_running_loss = 0.0
        # Loading weights in the model for eval
        vector_to_parameters(points[i, :], plain_model.parameters())
        for j, eval_data in enumerate(testloader, 0):
            eval_X, eval_Y = eval_data
            eval_X = eval_X.to(test_config['device'])
            eval_Y = eval_Y.to(test_config['device'])
            eval_outputs = plain_model(eval_X)

            loss = criterion(eval_outputs, eval_Y)
            eval_running_loss += loss.detach().item()
        # loss_ls.append(eval_running_loss/len(testloader))
        loss_ls.append(eval_running_loss)
        print(f"Point# {i}")
    # return torch.tensor(loss_ls).to(test_config['device']).view(len(loss_ls), 1)
    return {
        "loss": torch.tensor(loss_ls).to(test_config['device']).view(len(loss_ls), 1),
        "t": t_values
    }

### Curve eval on a straight line, END
