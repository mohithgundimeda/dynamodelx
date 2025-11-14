import torch
import inspect
from torch.optim import Optimizer
from typing import Literal, Dict, Type

OPTIMIZER_MAP: Dict[str, Type[Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD
}

OptimizerType = Literal["adam", "adamw", "sgd"]

def validate_optimizer(optimizer: str) -> str:
    """
    Validates the user prefered optimization function
    Raises ValueError and TypeError if invalid.
    """

    if not isinstance(optimizer, str):
        raise TypeError(f'Expected optimizer to be a string, but got {type(optimizer)}')

    optimizer = optimizer.lower().strip()

    if optimizer not in OPTIMIZER_MAP:
        raise ValueError(f"Invalid optimizer '{optimizer}'. Choose from: {list(OPTIMIZER_MAP.keys())}")
    
    return optimizer
    
def get_optimizer(optimizer: str, **kwargs) -> Optimizer:
    """
    Returns the optimizer instance for the given optimizer name.
    """
    optimizer_class = OPTIMIZER_MAP[optimizer]
    valid_params = inspect.signature(optimizer_class).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return optimizer_class(**filtered_kwargs)
