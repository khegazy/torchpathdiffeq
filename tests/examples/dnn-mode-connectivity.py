import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import torchvision
import torchvision.transforms as transforms
import numpy as np
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
python3 dnn-mode-connectivity.py -rd 20 -tc False
python dnn-mode-connectivity.py -d mnist
"""


parser = argparse.ArgumentParser(
                    prog='DNN-mode-connectivity',
                    description='Constructs constant-accuracy curves between independent DNN modes',
                    epilog='Sucks to suck')
parser.add_argument('-d', '--dataset')
parser.add_argument('-rd', '--resnet_depth')
parser.add_argument('-p', '--cifar10_path')
parser.add_argument('-w1', '--w_init')
parser.add_argument('-w2', '--w_end')
parser.add_argument('-mp', '--model_path')
parser.add_argument('-tc', '--train_curve')


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
    'w1_path': "./checkpoints_init_mlp_epochs_100LR_0.001.pt",
    'w2_path': "./checkpoints_end_MLP_epochs_100LR_0.001.pt",
    # 'w1_path': "./checkpointsinit_resnet_20_epochs_100LR_0.001.pt",
    # 'w2_path': "./checkpointsend_resnet_20_epochs_100LR_0.001.pt",
    'curve_path': "./mode_test_path_t_detached.pt"
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

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNISTMLP(nn.Module):
    ### 2-layer MLP for solving MNIST ###
    def __init__(self):
        super(MNISTMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (784) to hidden layer (128)
        self.fc2 = nn.Linear(128, 64)        # Hidden layer (128) to hidden layer (64)
        self.fc3 = nn.Linear(64, 10)         # Hidden layer (64) to output layer (10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))  # First hidden layer with ReLU activation
        x = F.relu(self.fc2(x))  # Second hidden layer with ReLU activation
        x = self.fc3(x)          # Output layer (logits)
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
        self.nn_dims = nn_dims
        # self.model = model #TODO: Keeping the initial model on hand for evaluation... or not?

    def forward(self, t): # `t\in [0, 1]`
        #TODO: enforce `t\in [0, 1]` --- this will be respected by `get_parallel_RK_solver`, confirm
        #TODO: How do I enforce CurveNet(0) = w1 && CurveNet(1) = w2?
        #TODO: How will a `CurveNet` instance actually be trained?
        coeffs = [1 - t, t, 1 - torch.cos(2 * math.pi * t)]
        t = F.tanh(self.fc1(t))
        t = F.tanh(self.fc2(t))
        t = F.tanh(self.fc3(t)) # `t` now represents a point in weight-space

        # enforcing that `Phi_t` evaluates to w_1 @ 0 and w_2 @ 1
        if len(t.shape) < 2:
            return (coeffs[0] * self.w1 + coeffs[1] * self.w2 + coeffs[2] * t).view(1, self.nn_dims)
        else:
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

    def forward(self, t):#, X, Y):
        w = self.curve(t) # grab the curves output
        loss_ls = []
        self.model.eval()
        for dim in range(w.shape[0]):
            eval_running_loss = 0.0
            vector_to_parameters(w[dim, :].view(-1), self.model.parameters())

            for i, eval_data in enumerate(testloader, 0):
                eval_X, eval_Y = eval_data
                eval_X = eval_X.to(test_config['device'])
                eval_Y = eval_Y.to(test_config['device'])
                eval_outputs = self.model(eval_X)

                loss = self.criterion(eval_outputs, eval_Y)
                eval_running_loss += loss.detach().item()
            loss_ls.append(eval_running_loss)
        return torch.tensor(loss_ls).to(test_config['device']).view(len(loss_ls), 1)

        # for dim in range(w.shape[0]):
        #     vector_to_parameters(w[dim, :].view(-1), self.model.parameters())
        #     loss.append(self.criterion(self.model(X), Y).view(1,1))
        # return torch.tensor(loss).to(test_config['device']).view(len(loss), 1)
### Describing the network used for the path ###


### Data loading ###
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

trainloader, testloader = load_data()
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


def train_mlp_mnist(optimizer_str: str, criterion_str: str, init_point: bool = True):
    criterion = test_config['criterion'][criterion_str]

    # net = resnet(num_classes=10, depth=test_config['resnet_depth']).to(test_config['device'])
    net = MNISTMLP()
    # Xavier initialization for the input layer to the ResNet:
    nn.init.xavier_uniform_(net.fc1.weight)

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
        test_config['model_init_path'] += test_config['checkpoints_path'] + "_init_MLP_" + "epochs_" + f"{test_config['epochs']}"\
                + "LR_" + f"{test_config['LR']}.pt"
    else:
        test_config['model_end_path'] += test_config['checkpoints_path'] + "_end_MLP_" + "epochs_" + f"{test_config['epochs']}"\
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
    print(f'{optimizer_str}')
    print(f'{test_config["device"]}')

    integral = None

    for epoch in range(test_config['epochs']):
        path.train()
        optimizer.zero_grad()

        integral = integrator.integrate(
            potential, t_init=t_init, t_final=t_final, max_batch=test_config['max_batches']
        )

        if test_config['max_batches'] is None:
            integral.integral.backward()

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
    torch.save({
                'epoch': test_config["epochs"],
                'model_state_dict': path.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'mode_test_path_t_detached.pt')
    torch.save(integral.integral, 'mode_test_integral_t_detached_correct.pt')
    torch.save(integral.loss, 'mode_test_loss_t_detached_correct.pt')
    torch.save(integral.t_pruned, 'mode_test_t_pruned_t_detached_correct.pt')
    torch.save(integral.t, 'mode_test_t_t_detached_correct.pt')
    torch.save(integral.h, 'mode_test_h_t_detached_correct.pt')
    torch.save(integral.y, 'mode_test_y_t_detached_correct.pt')
    torch.save(integral.sum_steps, 'mode_test_sum_steps_t_detached_correct.pt')
    torch.save(integral.integral_error, 'mode_test_integral_error_t_detached_correct.pt')
    torch.save(integral.errors, 'mode_test_errors_t_detached_correct.pt')
    torch.save(integral.error_ratios, 'mode_test_error_ratios_t_detached_correct.pt')
### Path training loop ###

### Path eval ###
def eval_curve(curve: CurveNet, potential: CurveNetLoss, t: torch.Tensor, og_model: nn.Module, tmp: bool):
    criterion = test_config['criterion']['CE']
    ls_list = []

    for epoch in range(test_config['epochs']):
        test_running_loss = 0.0
        curve.eval()
        for test_i, test_data in enumerate(testloader, 0):
            test_inputs, test_labels = test_data
            test_inputs  = test_inputs.to(test_config['device'])
            test_labels  = test_labels.to(test_config['device'])
            test_outputs = og_model(test_inputs)

            # loss = potential(t, test_inputs, test_labels)
            loss = criterion(test_outputs, test_labels)
            test_running_loss += loss.detach().item()

        ls_list.append(test_running_loss/len(testloader))
        print(f"Epoch# {epoch}")

    if tmp:
        torch.save(torch.tensor(ls_list), "resnet_w1_eval_with_state_dict.pt")
    else:
        torch.save(torch.tensor(ls_list), "resnet_w1_eval_v2p.pt")

    return ls_list
### Path eval ###

# train_mlp_mnist('Adam', 'CE')
# train_mlp_mnist('Adam', 'CE', False)

if args.train_curve is None:
    parallel_integrator = get_parallel_RK_solver(
        test_config['sampling_type'], method=test_config['method'], atol=test_config['atol'],\
        rtol=test_config['rtol'], remove_cut=0.1, dtype=torch.float
    ) # instantiating the integrator

    t_init = torch.tensor([0]).to(test_config['device'])
    t_final = torch.tensor([1]).to(test_config['device'])

    # This is what we will load weights into and use for evaluating the loss
    if test_config['dataset'] == 'mnist':
        plain_model = MNISTMLP()
    else:
        plain_model = resnet(num_classes=10, depth=test_config['resnet_depth']).to(test_config['device'])
    model_w1 = torch.load(test_config['w1_path'], map_location=test_config['device']) # we have `epoch`, `model_state_dict` and..
    model_w2 = torch.load(test_config['w2_path'], map_location=test_config['device']) # ... `optimizer_state_dict` information

    # names of subset of parameters to grab from state_dict
    param_names = {name for name, _ in plain_model.named_parameters()}

    model_w1_params = [param for name, param in model_w1['model_state_dict'].items() if name in param_names]
    w1 = parameters_to_vector(model_w1_params) # flattened params

    model_w2_params = [param for name, param in model_w2['model_state_dict'].items() if name in param_names]
    w2 = parameters_to_vector(model_w2_params) # flattened params

    path = CurveNet(w2.shape[0], w1, w2, test_config['criterion']['CE']).to(test_config['device']) # path NN
    path_loss = CurveNetLoss(test_config['criterion']['CE'], plain_model, path).to(test_config['device'])# potential along path

    train_path('Adam', path, path_loss, parallel_integrator)
else:
    t_init = torch.tensor([0], dtype=torch.float).to(test_config['device'])
    t_final = torch.tensor([1], dtype=torch.float).to(test_config['device'])

    plain_model = resnet(num_classes=10, depth=test_config['resnet_depth']).to(test_config['device'])
    plain_model_ = resnet(num_classes=10, depth=test_config['resnet_depth']).to(test_config['device'])
    model_w1 = torch.load(test_config['w1_path'], map_location=test_config['device'])
    model_w2 = torch.load(test_config['w2_path'], map_location=test_config['device'])

    param_names = {name for name, _ in plain_model.named_parameters()}

    # Filter checkpoint's state_dict to only include these parameters
    model_w1_params = [param for name, param in model_w1['model_state_dict'].items() if name in param_names]
    # model_w1_params = [tup[1] for tup in list(model_w1['model_state_dict'].items())]
    w1 = parameters_to_vector(model_w1_params)

    # model_w2_params = [tup[1] for tup in list(model_w2['model_state_dict'].items())]
    model_w2_params = [param for name, param in model_w2['model_state_dict'].items() if name in param_names]
    w2 = parameters_to_vector(model_w2_params)

    path = CurveNet(w2.shape[0], w1, w2, test_config['criterion']['CE']).to(test_config['device'])
    # path.load_state_dict(torch.load(test_config['curve_path'])['model_state_dict']) # loading path ckpt

    path_loss = CurveNetLoss(test_config['criterion']['CE'], plain_model, path).to(test_config['device'])

    # print(path(t_init).view(-1) == w1)
    # plain_model.load_state_dict(model_w1['model_state_dict'])
    # vector_to_parameters(path(t_init).view(-1), plain_model.parameters())
    vector_to_parameters(w1, plain_model.parameters())
    plain_model_.load_state_dict(model_w1['model_state_dict'])
    # eval_curve(path, path_loss, t_init, plain_model, False)
    # eval_curve(path, path_loss, t_init, plain_model_, True)

    # for p1, p2 in zip(plain_model.parameters(), plain_model_.parameters()):
    #     if p1.data.ne(p2.data).sum() > 0:
    #         print(False)
    # print(True)


    # ls_list = eval_curve(path, path_loss, t_final, plain_model)
    pass
