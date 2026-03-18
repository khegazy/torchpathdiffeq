from __future__ import annotations


class BaseODE:
    def __init__(self, N_dims):
        self.N_dims = N_dims

    def __call__(self, t):
        raise NotImplementedError


class linear(BaseODE):
    def __init__(self):
        super().__init__(1)

    def __call__(self, t):
        return t


class quadratic(BaseODE):
    def __init__(self):
        super().__init__(1)

    def __call__(self, t):
        return t**2
