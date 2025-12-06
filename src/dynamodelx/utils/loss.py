import torch
from typing import Literal, Dict , Callable

LossType = Literal['mean_square_loss', 'binary_cross_entropy', 'cross_entropy_loss', 'gaussian_nll_loss']
bce_logits_loss = torch.nn.BCEWithLogitsLoss()

def N_ELBO_NLL(y_pred_mean: torch.Tensor, y_pred_std: torch.Tensor, y_true:torch.Tensor):
    var = torch.clamp(y_pred_std**2, min=1e-6)
    return torch.mean(0.5 * (((y_pred_mean - y_true)**2)/(var)) + 0.5 * torch.log(2*torch.pi*(var)))
    
def BCE(y_train_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
    return bce_logits_loss(y_train_pred, y_true)

def N_ELBO_KLD(mu, log_var):
    var = log_var.exp()
    return 0.5 * torch.sum(mu**2 + var - log_var - 1)

def N_ELBO_KLD_STD(mu, log_std):
    std = torch.exp(log_std)
    var = std**2
    return 0.5 * torch.sum(mu**2 + var - torch.log(var) - 1)
    
def GaussianNLLLoss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    mean, rho = torch.chunk(y_pred, 2, dim=1)
    
    std = torch.nn.functional.softplus(rho).clamp(min=1e-3)
    var = std ** 2 + 1e-6

    return torch.mean(
        0.5 * torch.log(2 * torch.pi * var) +
        0.5 * ((y_true - mean) ** 2) / var
    )



LOSS_MAP : Dict[str, torch.nn.Module | Callable] = {
    'mean_square_loss' : torch.nn.MSELoss(),
    'binary_cross_entropy' : torch.nn.BCEWithLogitsLoss(),
    'cross_entropy_loss' : torch.nn.CrossEntropyLoss(),
    'gaussian_nll_loss' : GaussianNLLLoss
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