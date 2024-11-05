import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import torchvision
import torchvision.transforms as transforms
from resnet import *
from torchpathdiffeq import\
    steps,\
    get_parallel_RK_solver,\
    UNIFORM_METHODS,\
    VARIABLE_METHODS\


from functools import partial
import os
import argparse

torch.set_default_dtype(torch.float)

"""
python dnn-mode-connectivity.py -rd 20 -p /.data -w1 ./checkpointsinit_resnet_20_epochs_100LR_0.001.pt\
-w2 checkpointsend_resnet_20_epochs_100LR_0.001.pt -mp ./init_model.pt
"""


parser = argparse.ArgumentParser(
                    prog='DNN-mode-connectivity',
                    description='Constructs constant-accuracy curves between independent DNN modes',
                    epilog='Sucks to suck')

parser.add_argument('-rd', '--resnet_depth')
parser.add_argument('-p', '--cifar10_path')
parser.add_argument('-w1', '--w_init')
parser.add_argument('-w2', '--w_end')
parser.add_argument('-mp', '--model_path')


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
    'batch_size': 32,
    'criterion': {
        'CE': nn.CrossEntropyLoss()
    },
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'LR': 1e-3,
    'resnet_depth': int(args.resnet_depth),
    'cifar10_path': './data',
    'checkpoints_path': './checkpoints',
    'sampling_type': steps.ADAPTIVE_UNIFORM,
    'method': 'adaptive_heun',
    'atol': 1e-9,
    'rtol': 1e-7,
    'max_batches': 512,
    'model_init_path': "",
    'model_end_path': "",
    'w1_path': "./checkpointsinit_resnet_20_epochs_100LR_0.001.pt",
    'w2_path': "./checkpointsend_resnet_20_epochs_100LR_0.001.pt"
}

test_config['optims'] = {
        'Adam': partial(optim.Adam, lr=test_config["LR"])
}

### config dictionary for the tests ###


### Test CNN architecture ###
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
### Test CNN architecture ###


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
        # self.model = model #TODO: Keeping the initial model on hand for evaluation... or not?

    def forward(self, t): # `t\in [0, 1]`
        #TODO: enforce `t\in [0, 1]` --- this will be respected by `get_parallel_RK_solver`, confirm
        #TODO: How do I enforce CurveNet(0) = w1 && CurveNet(1) = w2?
        #TODO: How will a `CurveNet` instance actually be trained?
        coeffs = [1 - t, t, 1 - torch.cos(2 * math.pi * t)]
        #MARK: Data types mismatching between `t` and weights in `self.fc1`
        t = F.tanh(self.fc1(t))
        t = F.tanh(self.fc2(t))
        t = F.tanh(self.fc3(t)) # `t` now represents a point in weight-space

        # enforcing that `Phi_t` evaluates to w_1 @ 0 and w_2 @ 1
        return coeffs[0] * self.w1 + coeffs[1] * self.w2 + coeffs[2] * t

class CurveNetLoss(nn.Module):
    """
    Computes the loss L(w) := L(Y, model(X; w))
    Should be passed to the integrator
    """
    def __init__(self, criterion: nn.Module, model: nn.Module, curve: CurveNet):
        super(CurveNetLoss, self).__init__()
        self.criterion = criterion
        self.model = model # we don't need the trained weights here, just require the class itself
        self.curve = curve

    def forward(self, t, X, Y):
        print("actual_input ", t.shape)
        w = self.curve(t) # grab the curves output
        loss = []
        for dim in range(w.shape[0]):
            vector_to_parameters(w[dim, :].view(-1), self.model.parameters())
            #MARK: Data type discrepancy @ self.model(X)
            # i.e. between the inputs and the weights of the models
            # i.e. between the output of `CurveNet` (doubles) and the inputs (floats)
            loss.append(self.criterion(self.model(X), Y).view(1,1))
        return torch.tensor(loss).to(test_config['device']).view(len(loss), 1)
### Describing the network used for the path ###



### Data loading ###
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=test_config['cifar10_path'], train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=test_config['batch_size'],
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=test_config['cifar10_path'], train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_config['batch_size'],
                                         shuffle=False, num_workers=2)
### Data loading ###



def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()


### End-points training loop ###
def train_resnet_cifar10(optimizer_str: str, criterion_str: str, model: str = 'resnet', init_point: bool = True):
    criterion = test_config['criterion'][criterion_str]

    if model == 'cnn':
        net = Net().to(test_config['device'])
        # Xavier initialization for the input layer to the CNN:
        nn.init.xavier_uniform_(net.conv1.weight)
    else:
        net = resnet(num_classes=10, depth=test_config['resnet_depth']).to(test_config['device'])
        # Xavier initialization for the input layer to the ResNet:
        nn.init.xavier_uniform_(net.conv1.weight)

    optimizer = test_config['optims'][optimizer_str](net.parameters())

    print(f'{optimizer_str}')
    print(f'{test_config["device"]}')

    ls_list = []
    acc_list = []
    for epoch in range(test_config['epochs']):
        train_running_loss = 0.0
        train_acc          = 0.0

        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs         = inputs.to(test_config['device'])
            labels         = labels.to(test_config['device'])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss    = criterion(outputs, labels) # + (1e-2)*sum([torch.norm(p) for p in net.parameters()])
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc          += get_accuracy(outputs, labels, test_config['batch_size'])

        net.eval()
        test_acc = 0.0
        for test_i, test_data in enumerate(testloader, 0):
            test_inputs, test_labels = test_data
            test_inputs  = test_inputs.to(test_config['device'])
            test_labels  = test_labels.to(test_config['device'])
            test_outputs = net(test_inputs)
            test_acc    += get_accuracy(test_outputs, test_labels, test_config['batch_size'])
        print(f"Test Accuracy: {test_acc/len(testloader)}")
        acc_list.append(test_acc/len(testloader))

        print(f'Epoch: {epoch} \t Loss: {train_running_loss/len(trainloader)}')
        ls_list.append(train_running_loss/len(trainloader))

    if init_point:
        test_config['model_init_path'] += test_config['checkpoints_path'] + "init_resnet_" + f"{test_config['resnet_depth']}_" + "epochs_" + f"{test_config['epochs']}"\
                + "LR_" + f"{test_config['LR']}.pt"
    else:
        test_config['model_end_path'] += test_config['checkpoints_path'] + "end_resnet_" + f"{test_config['resnet_depth']}_" + "epochs_" + f"{test_config['epochs']}"\
                + "LR_" + f"{test_config['LR']}.pt"

    torch.save({
                'epoch': test_config["epochs"],
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, test_config['model_init_path'] if init_point else test_config['model_end_path'])

    print('Finished Training')
    print('____________________________________________________________')
    return ls_list, acc_list
### End-points training loop ###

### Path training loop ###
def train_path(optimizer_str: str, path: CurveNet, potential: CurveNetLoss, integrator):
    optimizer = test_config['optims'][optimizer_str](path.parameters())

    for epoch in range(test_config['epochs']):
        path.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs         = inputs.to(test_config['device'])
            labels         = labels.to(test_config['device'])

            optimizer.zero_grad()

            integrator.integrate(
                potential, t_init=t_init, t_final=t_final, max_batch=test_config['max_batches'], ode_args=(inputs, labels)
            )

            if integrator._integrator.max_batch is None:
                integrator.integral.backward()

            optimizer.step()
### Path training loop ###

parallel_integrator = get_parallel_RK_solver(
    test_config['sampling_type'], method=test_config['method'], atol=test_config['atol'],\
    rtol=test_config['rtol'], remove_cut=0.1, dtype=torch.float
) # instantiating the integrator

t_init = torch.tensor([0])
t_final = torch.tensor([1])

# This is what we will load weights into and use for evaluating the loss
plain_model = resnet(num_classes=10, depth=test_config['resnet_depth']).to(test_config['device'])
model_w1 = torch.load(test_config['w1_path'], map_location=test_config['device']) # we have `epoch`, `model_state_dict` and..
model_w2 = torch.load(test_config['w2_path'], map_location=test_config['device']) # ... `optimizer_state_dict` information

model_w1_params = [tup[1] for tup in list(model_w1['model_state_dict'].items())]
w1 = parameters_to_vector(model_w1_params) # flattened params

model_w2_params = [tup[1] for tup in list(model_w2['model_state_dict'].items())]
w2 = parameters_to_vector(model_w2_params) # flattened params


path = CurveNet(w2.shape[0], w1, w2, test_config['criterion']['CE']).to(test_config['device']) # path NN
path_loss = CurveNetLoss(test_config['criterion']['CE'], plain_model, path).to(test_config['device']) # potential along path

train_path('Adam', path, path_loss, parallel_integrator)
