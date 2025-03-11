import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import matplotlib.pyplot as plt
import torchvision
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


parser = argparse.ArgumentParser(
                    prog='DNN-mode-connectivity',
                    description='Constructs constant-accuracy curves between independent DNN modes',
                    epilog='Sucks to suck')
parser.add_argument('-w1', '--w_init')
parser.add_argument('-w2', '--w_end')
parser.add_argument('-mp', '--model_path')
parser.add_argument('-tc', '--train_curve')


args = parser.parse_args()


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
