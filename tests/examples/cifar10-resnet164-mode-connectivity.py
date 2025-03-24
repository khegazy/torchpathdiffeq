import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torch.func import vmap
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision
import numpy as np
from model import resnet_164
from torchpathdiffeq import\
    steps,\
    get_parallel_RK_solver,\
    UNIFORM_METHODS,\
    VARIABLE_METHODS


from functools import partial
import os
import argparse


CIFAR10_DIR = './data/'
MNIST_DIR = './data'
WORKERS = 4
BATCH_SIZE = 128

parser = argparse.ArgumentParser(
                    prog='DNN-mode-connectivity',
                    description='Constructs constant-accuracy curves between independent DNN modes',
                    epilog='Sucks to suck')

parser.add_argument('w_init', type=str, help='Path to initial weights')
parser.add_argument('w_end', type=str, help='Path to end weights')
parser.add_argument('experiment_id', type=int, help='Experiment ID')
parser.add_argument('seed', type=int, help='Random seed')

args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# python cifar10-resnet164-mode-connectivity.py -w1 "./MNIST_MLP_id=1.pt" -w2 "./MNIST_MLP_id=2.pt" -id 1 -s 42
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# train_set = CIFAR10(root=CIFAR10_DIR, train=True, transform=transforms.Compose([
#                         transforms.RandomHorizontalFlip(),
#                         transforms.RandomCrop((32, 32), 4),
#                         transforms.ToTensor(), normalize]))

# trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
#                               num_workers=WORKERS, pin_memory=True)

# testloader = DataLoader(CIFAR10(root=CIFAR10_DIR, train=False, transform=
#                                     transforms.Compose([
#                                         transforms.ToTensor(), normalize])),
#                                 batch_size=BATCH_SIZE, shuffle=False,
#                                 num_workers=WORKERS, pin_memory=True)


# if args.w_init is not None and args.w_end is None:
#     if args.model_path is None:
#         raise ValueError("Please specify both ends of the curve in addition to the model path")
#     raise ValueError("Both ends of the curve should be specified")

# if args.w_init is None and args.w_end is not None:
#     if args.model_path is None:
#         raise ValueError("Please specify both ends of the curve in addition to the model path")
#     raise ValueError("Both ends of the curve should be specified")


### config dictionary for the tests ###
test_config = {
    'epochs': 100,
    'epochs_integrator': 100,
    'batch_size': 32,
    'criterion': {
        'CE': nn.CrossEntropyLoss()
    },
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    # 'dataset': str(args.dataset),
    'LR': 1e-3,
    'cifar10_path': './data',
    'checkpoints_path': './checkpoints',
    'sampling_type': steps.ADAPTIVE_UNIFORM,
    'method': 'adaptive_heun',
    'dataset': 'mnist',
    'atol': 1e-5,
    'rtol': 1e-3,
    'max_batches': 256,
    'model_init_path': "",
    'model_end_path': "",
    'w1_path': args.w_init,
    'w2_path': args.w_end,
    'curve_path': "./mode_test_path_t_detached.pt",
    'id': int(args.experiment_id),
    'seed': int(args.seed)
}

test_config['optims'] = {
        'Adam': partial(optim.Adam, lr=test_config["LR"])
}
test_config['t_init'] = torch.tensor([0]).to(test_config['device']).type(torch.float)
test_config['t_final'] = torch.tensor([1]).to(test_config['device']).type(torch.float)


def load_data():
    if test_config['dataset'] == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        return (trainloader, testloader)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # Ratio of the dataset to use
        ratio = 0.05  # for example, 10% of the dataset

        # Load the full trainset and testset
        full_trainset = torchvision.datasets.CIFAR10(root=test_config['cifar10_path'], train=True,
                                                    download=True, transform=transform)
        full_testset = torchvision.datasets.CIFAR10(root=test_config['cifar10_path'], train=False,
                                                    download=True, transform=transform)

        # Calculate the number of samples to include
        train_size = int(len(full_trainset) * ratio)
        test_size = int(len(full_testset) * ratio)

        # Randomly select indices for subset
        train_indices = np.random.choice(len(full_trainset), train_size, replace=False)
        test_indices = np.random.choice(len(full_testset), test_size, replace=False)

        # Create subsets of the trainset and testset
        trainset = torch.utils.data.Subset(full_trainset, train_indices)
        testset = torch.utils.data.Subset(full_testset, test_indices)

        # Create DataLoaders with the subset
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=test_config['batch_size'],
                                                shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_config['batch_size'],
                                         shuffle=False, num_workers=2)

        return (trainloader, testloader)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

testloader, trainloader = load_data()

### Describing the network & Loss used for the path, BEGIN ###

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
        f1 = F.tanh(self.fc3(F.tanh(self.fc2(F.tanh(self.fc1(test_config['t_final']))))))
        f0 = F.tanh(self.fc3(F.tanh(self.fc2(F.tanh(self.fc1(test_config['t_init']))))))
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

class CurveNetLoss_vmap(nn.Module):
    """
    Computes the loss L(w) := L(Y, model(X; w))
    Should be passed to the integrator
    """
    def __init__(self, criterion: nn.Module, model: nn.Module, curve: CurveNet):
        super(CurveNetLoss_vmap, self).__init__()
        self.criterion = criterion
        self.model = model # we don't need the trained weights here, just require the class itself
        self.curve = curve
        self._threads = 10

        self._models: list[nn.Module] = [self.model] * self._threads

    def forward(self, t):#, X, Y):
        #TODO: pass in the weights here instead of time
        w = self.curve(t) # grab the curves output
        loss_ls = []
        self.model.eval()
        for dim in range(w.shape[0]):
            eval_running_loss = 0.0
            # start_time = time.time()
            vector_to_parameters(w[dim, :].view(-1), self.model.parameters())
            # end_time = time.time()
            # execution_time = end_time - start_time

            # print(f"Execution time: {execution_time:.6f} seconds")

            for i, eval_data in enumerate(testloader, 0):
                eval_X, eval_Y = eval_data
                eval_X = eval_X.to(test_config['device'])
                eval_Y = eval_Y.to(test_config['device'])
                eval_outputs = self.model(eval_X)

                loss = self.criterion(eval_outputs, eval_Y)
                eval_running_loss += loss.detach().item()
            loss_ls.append(eval_running_loss)
        return torch.tensor(loss_ls).to(test_config['device']).view(len(loss_ls), 1)

class CurveNetLoss_base(nn.Module):
    """
    Computes the loss L(w) := L(Y, model(X; w))
    Should be passed to the integrator
    """
    def __init__(self, criterion: nn.Module, model: nn.Module, curve: CurveNet):
        super(CurveNetLoss_base, self).__init__()
        self.criterion = criterion
        self.model = model # we don't need the trained weights here, just require the class itself
        self.curve = curve

    def forward(self, t):#, X, Y):
        #TODO: pass in the weights here instead of time
        w = self.curve(t) # grab the curves output
        loss_ls = []
        self.model.eval()
        for dim in range(w.shape[0]):
            eval_running_loss = 0.0
            # start_time = time.time()
            vector_to_parameters(w[dim, :].view(-1), self.model.parameters())
            # end_time = time.time()
            # execution_time = end_time - start_time

            # print(f"Execution time: {execution_time:.6f} seconds")

            for i, eval_data in enumerate(testloader, 0):
                eval_X, eval_Y = eval_data
                eval_X = eval_X.to(test_config['device'])
                eval_Y = eval_Y.to(test_config['device'])
                eval_outputs = self.model(eval_X)

                loss = self.criterion(eval_outputs, eval_Y)
                eval_running_loss += loss.detach().item()
            loss_ls.append(eval_running_loss)
        return torch.tensor(loss_ls).to(test_config['device']).view(len(loss_ls), 1)

class CurveNetLoss_torchmp(nn.Module):
    """
    Computes the loss L(w) := L(Y, model(X; w)), multithreaded
    Should be passed to the integrator
    NOTE: Need to verify thread-safety of this implementation
    """
    def __init__(self, criterion: nn.Module, model: nn.Module, curve: CurveNet):
        super(CurveNetLoss_torchmp, self).__init__()
        self.criterion = criterion
        self.model = model # we don't need the trained weights here, just require the class itself
        self.curve = curve

        #NOTE: Modify suitably for Perlmutter
        self._threads = 10

        self._models: list[nn.Module] = [self.model] * self._threads

    def worker(self, w, loss_ls, dim, thread_id):
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

    def forward(self, t):#, X, Y):
        #TODO: pass in the weights here instead of time
        w = self.curve(t) # grab the curves output
        models_per_thread: int = w.shape[0] / self._threads

        loss_ls = [0] * w.shape[0] # pre-allocating the loss array

        # Mapping between models and threads
        model_to_thread = [(w, loss_ls, dim, dim % models_per_thread) for dim in range(w.shape[0])]
        self.model.eval()


        mp.set_start_method('spawn')  # Use 'spawn' method for better compatibility
        with mp.Pool(processes=self._threads) as pool:
            # Pass an index (thread_id) along with parameters
            # pool.starmap(worker, [(i, w[dim, :], self._models[i]) for (dim, i) in model_to_thread])
            pool.starmap(self.worker, model_to_thread)

        return torch.tensor(loss_ls).to(test_config['device']).view(len(loss_ls), 1)

        # for dim in range(w.shape[0]):
        #     vector_to_parameters(w[dim, :].view(-1), self.model.parameters())
        #     loss.append(self.criterion(self.model(X), Y).view(1,1))
        # return torch.tensor(loss).to(test_config['device']).view(len(loss), 1)

### Describing the network & Loss used for the path, END ###

### Path training loop ###
def train_path(optimizer_str: str, path: CurveNet, potential: CurveNetLoss_base, integrator):
    optimizer = test_config['optims'][optimizer_str](path.parameters())
    print(f'{optimizer_str}')
    print(f'{test_config["device"]}')
    int_list = []

    integral = None

    for epoch in range(test_config['epochs_integrator']):

        path.train()
        optimizer.zero_grad()

        integral = integrator.integrate(
            potential, t_init=test_config['t_init'], t_final=test_config['t_final'], max_batch=test_config['max_batches']
        )

        if test_config['max_batches'] is None:
            integral.integral.backward()

        int_list.append(integral.integral)


        optimizer.step()
        print(f'Epoch#: {epoch}')

    # for epoch in range(test_config['epochs']):
    #     path.train()
    #     for i, data in enumerate(trainloader, 0):
    #         inputs, labels = data
    #         inputs         = inputs.to(test_config['device'])
    #         labels         = labels.to(test_config['device'])

    #         optimizer.zero_grad()


    #         integral = integrator.integrate(
    #             potential, t_init=t_init, t_final=t_final, max_batch=test_config['max_batches'], ode_args=(inputs, labels)
    #         )

    #         if test_config['max_batches'] is None:
    #             integral.integral.backward()

    #         optimizer.step()
        # print(f'Epoch#: {epoch}')
    # Saving all of the OP tensors
    int_list = torch.tensor(int_list)
    torch.save({
                'epoch': test_config["epochs"],
                'model_state_dict': path.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'mode_test_path_t_detached.pt')

    torch.save(int_list, f'mode_test_integral_all_id={test_config["id"]}.pt')
    torch.save(integral.integral, f'mode_test_integral_id={test_config["id"]}.pt')
    torch.save(integral.loss, f'mode_test_loss_id={test_config["id"]}.pt')
    torch.save(integral.t_pruned, f'mode_test_t_pruned_id={test_config["id"]}.pt')
    torch.save(integral.t, f'mode_test_t_id={test_config["id"]}.pt')
    torch.save(integral.h, f'mode_test_h_id={test_config["id"]}.pt')
    torch.save(integral.y, f'mode_test_y_id={test_config["id"]}.pt')
    torch.save(integral.sum_steps, f'mode_test_sum_stepid={test_config["id"]}.pt')
    torch.save(integral.integral_error, f'mode_test_integral_erroid={test_config["id"]}.pt')
    torch.save(integral.errors, f'mode_test_errors_id={test_config["id"]}.pt')
    torch.save(integral.error_ratios, f'mode_test_error_ratioid={test_config["id"]}.pt')

### Path training loop ###

### Path eval ###
def eval_curve(curve: CurveNet, plain_model: nn.Module, num_points: int):
    #, tmp: bool):
    t_values = torch.linspace(0, 1, num_points).view(num_points, 1)
    # points = torch.lerp(w1, w2, t_values.unsqueeze(1))
    points = curve(t_values)
    criterion = test_config['criterion']['CE']
    ls_list = []

    # for epoch in range(test_config['epochs_integrator']):
    print("Curve\n")
    for i in range(num_points):
        eval_running_loss = 0.0
        vector_to_parameters(points[i, :], plain_model.parameters())
        curve.eval()
        for j, eval_data in enumerate(testloader, 0):
            eval_X, eval_Y = eval_data
            eval_X = eval_X.to(test_config['device'])
            eval_Y = eval_Y.to(test_config['device'])
            eval_outputs = plain_model(eval_X)

            loss = criterion(eval_outputs, eval_Y)
            eval_running_loss += loss.detach().item()

        ls_list.append(eval_running_loss/len(testloader))
        print(f"Point# {i}")

    # if tmp:
    #     torch.save(torch.tensor(ls_list), "resnet_w1_eval_with_state_dict.pt")
    # else:
    #     torch.save(torch.tensor(ls_list), "resnet_w1_eval_v2p.pt")

    # return ls_list

    return {
        "loss": torch.tensor(ls_list).to(test_config['device']).view(len(ls_list), 1),
        "t": t_values
    }

### Path eval ###

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

def setup(seed: int):
    """
    Sets up everything for training/eval
    """""
    parallel_integrator = get_parallel_RK_solver(
        test_config['sampling_type'], method=test_config['method'], atol=test_config['atol'],\
        rtol=test_config['rtol'], remove_cut=0.1, dtype=torch.float
    ) # instantiating the integrator

    # plain_model = resnet_164(10, seed)
    plain_model = MLP()

    model_w1 = torch.load(test_config['w1_path'], map_location=test_config['device']) # we have `epoch`, `model_state_dict` and..
    model_w2 = torch.load(test_config['w2_path'], map_location=test_config['device']) # ... `optimizer_state_dict` information

    # names of subset of parameters to grab from state_dict
    param_names = {name for name, _ in plain_model.named_parameters()}

    model_w1_params = [param for name, param in model_w1.items() if name in param_names]
    w1 = parameters_to_vector(model_w1_params) # flattened params

    model_w2_params = [param for name, param in model_w2.items() if name in param_names]
    w2 = parameters_to_vector(model_w2_params) # flattened params

    return (parallel_integrator, plain_model, w1, w2)


if __name__ == '__main__':
    train = True

    parallel_integrator, plain_model, w1, w2 = setup(test_config['seed'])
    if train:
        path = CurveNet(w2.shape[0], w1, w2, test_config['criterion']['CE']).to(test_config['device']) # path NN
        path_loss = CurveNetLoss_base(test_config['criterion']['CE'], plain_model, path).to(test_config['device'])# potential along path
        train_path('Adam', path, path_loss, parallel_integrator)
    else:
        #TODO: re-write evaluation routine
        pass
    pass
