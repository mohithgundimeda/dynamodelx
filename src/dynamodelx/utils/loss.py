import torch
from typing import Literal, Dict , Callable

LossType = Literal['mean_square_loss', 'binary_cross_entropy', 'cross_entropy_loss', 'gaussian_nll_loss']

def GaussianNLLLoss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    mean, var = torch.chunk(y_pred, 2, dim=1)
    var = torch.nn.functional.softplus(var).clamp(min=1e-6, max=1e2)
    return torch.mean(0.5 * torch.log(2 * torch.pi * var) + 0.5 * ((y_true - mean) ** 2) / var)


LOSS_MAP : Dict[str, torch.nn.Module | Callable] = {
    'mean_square_loss' : torch.nn.MSELoss(),
    'binary_cross_entropy' : torch.nn.BCEWithLogitsLoss(),
    'cross_entropy_loss' : torch.nn.CrossEntropyLoss(),
    'gaussian_nll_loss' : GaussianNLLLoss()
}

def validate_loss(loss: str) -> str:
    """
    Takes input loss from the user, validates it, raises error if it's invalid
    """
    if not isinstance(loss, str):
        raise TypeError(f'Expected loss to be a string, but recieved {type(loss)}')
    
    loss_name = loss.lower().strip()

    if loss_name not in LOSS_MAP:
        raise ValueError(f'Expected loss functions to be one of {list(LOSS_MAP.keys())}')
    
    return loss_name, LOSS_MAP[loss_name]