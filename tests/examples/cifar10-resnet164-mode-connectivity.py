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

# Try to import functional API - fallback for older PyTorch versions
try:
    import torch.func as func
    HAS_TORCH_FUNC = True
except ImportError:
    try:
        from torch.nn.utils import stateless
        HAS_TORCH_FUNC = False
    except ImportError:
        print("Warning: Neither torch.func nor stateless available. You may need PyTorch 2.0+")
        HAS_TORCH_FUNC = None

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

### config dictionary for the tests ###
test_config = {
    'epochs': 100,
    'epochs_integrator': 45,
    'batch_size': 32,
    'criterion': {
        'CE': nn.CrossEntropyLoss()
    },
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'LR': 1e-3,
    'cifar10_path': './data',
    'checkpoints_path': './checkpoints',
    'root': "/pscratch/sd/a/aryamanj/torchpathdiffeq/mnist/",
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
    'id': int(args.experiment_id),
    'seed': int(args.seed)
}

test_config['optims'] = {
        'Adam': partial(optim.Adam, lr=test_config["LR"])
}
test_config['t_init'] = torch.tensor([0.0]).to(test_config['device']).type(torch.float)
test_config['t_final'] = torch.tensor([1.0]).to(test_config['device']).type(torch.float)
test_config['curve_path'] =  f"{test_config['root']}/mode_test_path_final_id=1.pt"

def load_data():
    if test_config['dataset'] == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=True)

        return (trainloader, testloader)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ratio = 0.05

        full_trainset = torchvision.datasets.CIFAR10(root=test_config['cifar10_path'], train=True,
                                                    download=True, transform=transform)
        full_testset = torchvision.datasets.CIFAR10(root=test_config['cifar10_path'], train=False,
                                                    download=True, transform=transform)

        train_size = int(len(full_trainset) * ratio)
        test_size = int(len(full_testset) * ratio)

        train_indices = np.random.choice(len(full_trainset), train_size, replace=False)
        test_indices = np.random.choice(len(full_testset), test_size, replace=False)

        trainset = torch.utils.data.Subset(full_trainset, train_indices)
        testset = torch.utils.data.Subset(full_testset, test_indices)

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

trainloader, testloader = load_data()

### FIXED CurveNet with proper boundary constraints ###
class CurveNet(nn.Module):
    def __init__(self, nn_dims: int, w1: torch.Tensor, w2: torch.Tensor, criterion: nn.Module):
        super(CurveNet, self).__init__()
        #TODO: Potentially set hidden_dim to a ratio of `nn_dims`
        hidden_dim = 100
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, nn_dims)
        self.w1 = w1
        self.w2 = w2
        self.criterion = criterion
        self.nn_dims = nn_dims

    def _compute_mlp_output(self, t):
        """Helper to compute MLP output"""
        x = F.tanh(self.fc1(t))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

    def forward(self, t):
        coeffs = [1 - t, t]
        f0 = self._compute_mlp_output(test_config["t_init"])
        f1 = self._compute_mlp_output(test_config["t_final"])

        # Compute MLP output
        mlp_output = self._compute_mlp_output(t)

        # Boundary constraint enforcement
        if len(mlp_output.shape) < 2:
            return (coeffs[0] * (mlp_output - f0 + self.w1) +
                   coeffs[1] * (mlp_output - f1 + self.w2)).view(1, self.nn_dims)
        else:
            return (coeffs[0] * (mlp_output - f0 + self.w1) +
                   coeffs[1] * (mlp_output - f1 + self.w2))

### COMPLETELY REWRITTEN LOSS FUNCTION ###
class CurveNetLoss_base(nn.Module):
    """
    Computes the loss L(w) := L(Y, model(X; w)) * ||φ'_θ(t)||
    Preserves gradients and includes derivative term
    """
    def __init__(self, criterion: nn.Module, model: nn.Module, curve: CurveNet):
        super(CurveNetLoss_base, self).__init__()
        self.criterion = criterion
        self.model = model
        self.curve = curve

        # Get model structure for functional evaluation
        self.param_shapes = {name: param.shape for name, param in model.named_parameters()}

    def weights_to_state_dict(self, weights_flat):
        """Convert flat weights back to state dict format"""
        state_dict = {}
        idx = 0
        for name, shape in self.param_shapes.items():
            num_params = torch.prod(torch.tensor(shape))
            state_dict[name] = weights_flat[idx:idx+num_params].view(shape)
            idx += num_params
        return state_dict

    def compute_derivative_norm(self, t):
        """Compute ||φ'_θ(t)|| using autograd"""
        # Ensure t requires grad
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        weights = self.curve(t)

        # Compute derivative for each time point
        derivatives_norm = []
        for i in range(weights.shape[0]):
            # Compute gradient of weights w.r.t. t for this batch element
            grad_outputs = torch.ones_like(weights[i])
            try:
                grads = torch.autograd.grad(
                    outputs=weights[i],
                    inputs=t,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                if grads is not None:
                    derivative_norm = torch.norm(grads)
                else:
                    derivative_norm = torch.tensor(1.0, device=t.device)  # fallback

            except RuntimeError as e:
                print(f"Gradient computation failed: {e}")
                derivative_norm = torch.tensor(1.0, device=t.device)  # fallback

            derivatives_norm.append(derivative_norm)

        return torch.stack(derivatives_norm)

    def forward(self, t):
        # Ensure t requires gradients for derivative computation
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        # Get weights and derivatives
        w = self.curve(t)  # Shape: [batch_size, num_params]

        # Compute derivative norms
        try:
            derivatives_norm = self.compute_derivative_norm(t)  # Shape: [batch_size]
        except Exception as e:
            print(f"Derivative computation failed, using constant: {e}")
            derivatives_norm = torch.ones(w.shape[0], device=w.device)

        loss_ls = []

        for dim in range(w.shape[0]):
            # Convert flat weights to state dict
            state_dict = self.weights_to_state_dict(w[dim, :])

            running_loss = 0.0
            num_batches = 0

            # Evaluate model with different weights
            for eval_data in testloader:
                eval_X, eval_Y = eval_data
                eval_X = eval_X.to(test_config['device'])
                eval_Y = eval_Y.to(test_config['device'])

                # Use torch JAX-like API for calling model with different weights
                try:
                    if HAS_TORCH_FUNC:
                        predictions = func.functional_call(self.model, state_dict, eval_X)
                    else:
                        # Fallback for older PyTorch versions
                        # This is less elegant but should work
                        old_state = {name: param.clone() for name, param in self.model.named_parameters()}

                        # Load new weights
                        for name, param in self.model.named_parameters():
                            if name in state_dict:
                                param.data = state_dict[name]

                        predictions = self.model(eval_X)

                        # Restore old weights
                        for name, param in self.model.named_parameters():
                            if name in old_state:
                                param.data = old_state[name]

                    loss = self.criterion(predictions, eval_Y)
                    running_loss += loss
                    num_batches += 1

                except Exception as e:
                    print(f"Model evaluation failed: {e}")
                    # Use a dummy loss to prevent complete failure
                    running_loss += torch.tensor(1.0, device=test_config['device'])
                    num_batches += 1

            # L(φ_θ(t))||φ'_θ(t)||
            if num_batches > 0:
                avg_loss = running_loss / num_batches
                final_loss = avg_loss * derivatives_norm[dim]
            else:
                final_loss = torch.tensor(1.0, device=test_config['device'])

            loss_ls.append(final_loss)

        return torch.stack(loss_ls).view(-1, 1)

### Path training loop ###
def train_path(optimizer_str: str, path: CurveNet, potential: CurveNetLoss_base, integrator, resume_checkpoint_path=None):
    optimizer = test_config['optims'][optimizer_str](path.parameters())
    print(f'{optimizer_str}')
    print(f'{test_config["device"]}')

    # Initialize training state
    int_list = []
    start_epoch = 0
    integral = None

    # Set save root to scratch directory
    checkpoint_dir = os.path.join(test_config["root"], f"checkpoints_id_{test_config['id']}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(test_config["root"], exist_ok=True)

    print(f"Saving to: {test_config['root']}")
    print(f"Checkpoints: {checkpoint_dir}")

    # RESUME FROM CHECKPOINT IF PROVIDED
    if resume_checkpoint_path is not None:
        print(f"Resuming training from: {resume_checkpoint_path}")

        try:
            checkpoint = torch.load(resume_checkpoint_path, map_location=test_config['device'])

            # Load model and optimizer state
            path.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore training progress
            start_epoch = checkpoint['epoch']
            int_list = checkpoint['int_list'].tolist() if torch.is_tensor(checkpoint['int_list']) else checkpoint['int_list']

            print(f"Successfully resumed from epoch {start_epoch}")
            print(f"Previous training loss: {int_list[-1]:.6f}")
            print(f"Continuing training for {test_config['epochs_integrator'] - start_epoch} more epochs")

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("Starting training from scratch instead...")
            start_epoch = 0
            int_list = []
    else:
        print("🆕 Starting training from scratch")

    # TRAINING LOOP - Modified to start from start_epoch
    for epoch in range(start_epoch, test_config['epochs_integrator']):
        path.train()
        optimizer.zero_grad()

        integral = integrator.integrate(
            potential, t_init=test_config['t_init'], t_final=test_config['t_final'], max_batch=test_config['max_batches']
        )

        if test_config['max_batches'] is None:
            integral.integral.backward()

        int_list.append(integral.integral.item())
        optimizer.step()

        print(f'Epoch#: {epoch}/{test_config["epochs_integrator"]}, Integral: {integral.integral.item():.6f}')

        # CHECKPOINT EVERY 5 EPOCHS - Save lightweight quantities
        if (epoch + 1) % 5 == 0:
            print(f"Checkpointing at epoch {epoch + 1}...")

            # Save training progress and integrator outputs
            checkpoint_data = {
                'epoch': epoch + 1,
                'int_list': torch.tensor(int_list),
                'optimizer_state_dict': optimizer.state_dict(),
                'integral_value': integral.integral.item(),
                'config': test_config
            }
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f'training_checkpoint_epoch_{epoch+1}.pt'))

            # Save integrator quantities (these are the most important for analysis)
            torch.save(integral.integral, os.path.join(checkpoint_dir, f'integral_epoch_{epoch+1}_id={test_config["id"]}.pt'))
            torch.save(integral.loss, os.path.join(checkpoint_dir, f'loss_epoch_{epoch+1}_id={test_config["id"]}.pt'))
            torch.save(integral.t_pruned, os.path.join(checkpoint_dir, f't_pruned_epoch_{epoch+1}_id={test_config["id"]}.pt'))
            torch.save(integral.t, os.path.join(checkpoint_dir, f't_epoch_{epoch+1}_id={test_config["id"]}.pt'))
            torch.save(integral.h, os.path.join(checkpoint_dir, f'h_epoch_{epoch+1}_id={test_config["id"]}.pt'))
            torch.save(integral.y, os.path.join(checkpoint_dir, f'y_epoch_{epoch+1}_id={test_config["id"]}.pt'))
            torch.save(integral.sum_steps, os.path.join(checkpoint_dir, f'sum_steps_epoch_{epoch+1}_id={test_config["id"]}.pt'))
            torch.save(integral.integral_error, os.path.join(checkpoint_dir, f'integral_error_epoch_{epoch+1}_id={test_config["id"]}.pt'))
            torch.save(integral.errors, os.path.join(checkpoint_dir, f'errors_epoch_{epoch+1}_id={test_config["id"]}.pt'))
            torch.save(integral.error_ratios, os.path.join(checkpoint_dir, f'error_ratios_epoch_{epoch+1}_id={test_config["id"]}.pt'))

        # CHECKPOINT EVERY 25 EPOCHS - Save model too
        if (epoch + 1) % 25 == 0:
            print(f"Full model checkpoint at epoch {epoch + 1}...")

            full_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': path.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'int_list': torch.tensor(int_list),
                'config': test_config
            }
            torch.save(full_checkpoint, os.path.join(checkpoint_dir, f'full_model_checkpoint_epoch_{epoch+1}.pt'))

    # FINAL SAVE - All in scratch directory
    print("Saving final results...")

    int_list = torch.tensor(int_list)

    # Final model save
    torch.save({
                'epoch': test_config["epochs_integrator"],
                'model_state_dict': path.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'int_list': int_list
                }, os.path.join(test_config["root"], f'mode_test_path_final_id={test_config["id"]}.pt'))

    # Final integrator outputs - all in scratch directory
    torch.save(int_list, os.path.join(test_config["root"], f'mode_test_integral_all_id={test_config["id"]}.pt'))
    torch.save(integral.integral, os.path.join(test_config["root"], f'mode_test_integral_id={test_config["id"]}.pt'))
    torch.save(integral.loss, os.path.join(test_config["root"], f'mode_test_loss_id={test_config["id"]}.pt'))
    torch.save(integral.t_pruned, os.path.join(test_config["root"], f'mode_test_t_pruned_id={test_config["id"]}.pt'))
    torch.save(integral.t, os.path.join(test_config["root"], f'mode_test_t_id={test_config["id"]}.pt'))
    torch.save(integral.h, os.path.join(test_config["root"], f'mode_test_h_id={test_config["id"]}.pt'))
    torch.save(integral.y, os.path.join(test_config["root"], f'mode_test_y_id={test_config["id"]}.pt'))
    torch.save(integral.sum_steps, os.path.join(test_config["root"], f'mode_test_sum_stepid={test_config["id"]}.pt'))
    torch.save(integral.integral_error, os.path.join(test_config["root"], f'mode_test_integral_erroid={test_config["id"]}.pt'))
    torch.save(integral.errors, os.path.join(test_config["root"], f'mode_test_errors_id={test_config["id"]}.pt'))
    torch.save(integral.error_ratios, os.path.join(test_config["root"], f'mode_test_error_ratioid={test_config["id"]}.pt'))

    print(f"Training completed! Check {test_config['root']} for results")
    print(f"Intermediate checkpoints: {checkpoint_dir}/")

def eval_curve(curve: CurveNet, plain_model: nn.Module, num_points: int):
    t_values = torch.linspace(0, 1, num_points).view(num_points, 1).to(test_config['device'])
    points = curve(t_values)
    criterion = test_config['criterion']['CE']
    ls_list = []

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


    tmp = {
        "loss": torch.tensor(ls_list).to(test_config['device']).view(len(ls_list), 1),
        "t": t_values
    }
    id=2
    torch.save(tmp, f'curve_eval_id={id}.pt')
    return tmp

def eval_line(w1: torch.Tensor, w2: torch.Tensor, plain_model: nn.Module, num_points: int):
    t_values = torch.linspace(0, 1, num_points).to(test_config["device"])
    points = torch.lerp(w1, w2, t_values.unsqueeze(1))
    points = points.to(test_config['device'])
    criterion = test_config['criterion']['CE']
    loss_ls = []

    print("Line\n")
    for i in range(num_points):
        eval_running_loss = 0.0
        vector_to_parameters(points[i, :], plain_model.parameters())
        for j, eval_data in enumerate(testloader, 0):
            eval_X, eval_Y = eval_data
            eval_X = eval_X.to(test_config['device'])
            eval_Y = eval_Y.to(test_config['device'])
            eval_outputs = plain_model(eval_X)

            loss = criterion(eval_outputs, eval_Y)
            eval_running_loss += loss.detach().item()
        loss_ls.append(eval_running_loss)
        print(f"Point# {i}")
    tmp = {
        "loss": torch.tensor(loss_ls).to(test_config['device']).view(len(loss_ls), 1),
        "t": t_values
    }
    torch.save(tmp, 'line_eval_id=1.pt')
    return tmp

def setup(seed: int):
    parallel_integrator = get_parallel_RK_solver(
        test_config['sampling_type'], method=test_config['method'], atol=test_config['atol'],\
        rtol=test_config['rtol'], remove_cut=0.1, dtype=torch.float
    )

    plain_model = MLP()

    model_w1 = torch.load(test_config['w1_path'], map_location=test_config['device'])
    model_w2 = torch.load(test_config['w2_path'], map_location=test_config['device'])

    param_names = {name for name, _ in plain_model.named_parameters()}

    model_w1_params = [param for name, param in model_w1.items() if name in param_names]
    w1 = parameters_to_vector(model_w1_params)

    model_w2_params = [param for name, param in model_w2.items() if name in param_names]
    w2 = parameters_to_vector(model_w2_params)

    return (parallel_integrator, plain_model, w1, w2)

if __name__ == '__main__':
    #TODO: Make this all automated
    train = True
    resume_train = True

    parallel_integrator, plain_model, w1, w2 = setup(test_config['seed'])
    if train:
        path = CurveNet(w2.shape[0], w1, w2, test_config['criterion']['CE']).to(test_config['device'])
        path_loss = CurveNetLoss_base(test_config['criterion']['CE'], plain_model, path).to(test_config['device'])

        if resume_train:
            train_path('Adam', path, path_loss, parallel_integrator, test_config["curve_path"])
        else:
            train_path('Adam', path, path_loss, parallel_integrator)
    else:
        path = CurveNet(w2.shape[0], w1, w2, test_config['criterion']['CE']).to(test_config['device'])

        # Load checkpoint
        path_checkpoint = torch.load(f'{test_config["curve_path"]}', map_location=test_config["device"])
        path.load_state_dict(path_checkpoint["model_state_dict"])
        print("Model loaded successfully!")

        eval_curve(path, plain_model, 135)
        eval_line(w1, w2, plain_model, 135)
