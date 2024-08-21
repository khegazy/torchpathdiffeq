import torch 
from torchpathdiffeq import UNIFORM_METHODS, VARIABLE_METHODS

for name, method in UNIFORM_METHODS.items():
    error_message = f"Tableau B coefficients to not sum to 1 for uniform method {name}"
    assert torch.abs(torch.sum(method.tableau.b) - 1.0) < 1e-7, error_message

def test_tableau():
    n_samples = 100
    for name, method_class in VARIABLE_METHODS.items():
        method = method_class()
        if method.order < 3:
            continue
        c_tensor = torch.concatenate(
            [
                torch.zeros((n_samples, 1, 1)),
                torch.linspace(0.01, 0.99, n_samples)[:,None,None],
                torch.ones((n_samples, 1, 1))
            ],
            dim=1
        )
        b_tensor, _ = method.tableau_b(c_tensor)
        error_message = f"Tableau B coefficients to not sum to 1 for variable method {name}"
        assert torch.all(torch.abs(torch.sum(b_tensor, dim=-1) - 1.0) < 1e-5), error_message