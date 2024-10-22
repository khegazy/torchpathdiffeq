import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import torchvision
import torchvision.transforms as transforms
from resnet import *


from functools import partial
import argparse


parser = argparse.ArgumentParser(
                    prog='DNN-mode-connectivity',
                    description='Constructs constant-accuracy curves between independent DNN modes',
                    epilog='Sucks to suck')

parser.add_argument('-rd', '--resnet_depth')
parser.add_argument('-p', '--cifar10_path')
parser.add_argument('-w1', '--w_init')
parser.add_argument('-w2', '--w_end')

args = parser.parse_args()

if args.w_init is not None and args.w_end is None:
    raise ValueError("Both ends of the curve should be specified")

if args.w_init is None and args.w_end is not None:
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
    'cifar10_path': args.cifar10_path,
    'checkpoints_path': './checkpoints'
}
test_config['optims'] = {
        'Adam': partial(optim.Adam, lr=test_config["LR"])
}

model_init_path = ""
model_end_path = ""
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



### Describing the flow ###
class CurveFlow(nn.Module):
    def __init__(self, nn_dims: int, w1: torch.Tensor, w2: torch.Tensor, criterion: nn.Module, model: nn.Module, family: str = 'poly'):
        super(CurveFlow, self).__init__()
        self.family = family
        self.linear = nn.Linear(nn_dims, 1)
        self.w1 = w1
        self.w2 = w2
        self.criterion = criterion
        self.model = model

    def forward(self, x, y):
        t_tilde = torch.rand(1)
        curve_eval: torch.Tensor

        if self.family == 'poly':
            curve_eval = poly_chain(self.linear, t_tilde, self.w1, self.w2)
        else:
            curve_eval = bezier_curve(self.Linear, t_tilde, self.w1, self.w2)

        # Loading the current parameters in the network
        vector_to_parameters(curve_eval, self.model.parameters())
        self.criterion(y, self.model(x)).backward()

        #TODO: return the flattened gradient wrt every parameter


def poly_chain(params, t, w1, w2):
    # The output of this function should be a valid set of NN parameters
    if 0 <= t < 0.5:
        return 2 * (t * params + (0.5 - t) * w1)
    else:
        return 2 * ((t - 0.5) * w2 + (1 - t) * params)

def bezier_curve(params, t, w1, w2):
    # The output of this function should be a valid set of NN parameters
    return (1 - t) ** 2 * w1 + 2 * t * (1 - t) * params + t ** 2 * w2

### Describing the flow ###



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



### Main training loop ###
def train_resnet_cifar10(optimizer_str: str, criterion_str: str, model: str = 'cnn', init_point: bool = True):
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
    for epoch in range(test_config['epochs']):  # loop over the dataset multiple times
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
        model_init_path += test_config['checkpoints_path'] + "init_resnet_" + f"{test_config['resnet_depth']}_" + "epochs_" + f"{test_config['epochs']}"\
                + "LR_" + f"{test_config['LR']}.pt"
    else:
        model_end_path += test_config['checkpoints_path'] + "end_resnet_" + f"{test_config['resnet_depth']}_" + "epochs_" + f"{test_config['epochs']}"\
                + "LR_" + f"{test_config['LR']}.pt"

    torch.save({
                'epoch': test_config["epochs"],
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_init_path if init_point else model_end_path)

    print('Finished Training')
    print('____________________________________________________________')
    return ls_list, acc_list
### Main training loop ###



if args.w_init is None:
    # training end-points were not passed in and have to be trained
    train_resnet_cifar10('Adam', 'CE', 'resnet', True)
    train_resnet_cifar10('Adam', 'CE', 'resnet', False)

    torch.load(model_init_path)
    torch.load(model_end_path)
    pass
else:
    # training end-points were passed in and will be directly used for
    # the curve-fitting procedure
    pass
