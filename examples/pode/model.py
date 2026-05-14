from __future__ import annotations

import torch
from torch import nn


class DenseNet(nn.Module):
    def __init__(
        self,
        t_init,
        initial_condition,
        layers,
        N_output_dims,
        device,
        dtype=torch.float64,
        activation=nn.GELU(),
        output_activation=None,
        normalize=False,
    ):
        super().__init__()

        self.initial_condition = initial_condition
        self.t_init = t_init.to(device)

        self.n_layers = len(layers)
        assert self.n_layers >= 1
        layers.append(N_output_dims)
        self.device = device
        self.dtype = dtype
        self.activation = activation
        # self.activation = nn.GELU()
        # self.activation = nn.Tanh()
        # self.activation = nn.ELU()

        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(
                nn.Linear(
                    layers[i],
                    layers[i + 1],
                    dtype=dtype,
                    bias=False,
                    device=self.device,
                )
            )
            if i != self.n_layers - 1:
                if normalize:
                    raise NotImplementedError
                    self.layers.append(nn.BatchNorm1d(layers[i + 1], dtype=dtype))
                self.layers.append(self.activation)

        if output_activation is not None:
            self.layers.append(nn.GELU())

    def forward(self, x):
        x = x - self.t_init
        x = torch.movedim(x, 1, -1)
        for l in self.layers:
            x = l(x)
        x = torch.movedim(x, -1, 1)
        return x + self.initial_condition
