import torch
from typing import Literal, Dict, Callable, Optional

WeightInitType = Literal['xavier', 'he', 'uniform', 'normal']

INIT_MAP: Dict[str, Callable] = {
    'xavier': torch.nn.init.xavier_uniform_,
    'he': torch.nn.init.kaiming_uniform_,
    'normal': torch.nn.init.normal_,
    'uniform': torch.nn.init.uniform_,
}


def validate_weights_init(weights_init: Optional[str]) -> Optional[str]:
    """
    Validates the user-preferred weight initialization.
    Raises ValueError or TypeError if invalid.
    """
    if isinstance(weights_init, str):
        weights_init = weights_init.lower().strip()
        if weights_init not in INIT_MAP:
            raise ValueError(
                f'Invalid weight initialization "{weights_init}". '
                f'Choose from {list(INIT_MAP.keys())}.'
            )
    elif weights_init is not None:
        raise TypeError(
            f'Expected weights_init to be a string or None, got {type(weights_init)}.'
        )
    return weights_init


def _init_weights(model: torch.nn.Module, weights_init: Optional[str]) -> None:
    """
    Initializes Linear layer weights in the model using the given initialization scheme.
    If None, uses PyTorch's default initialization.
    """
    if weights_init is None:
        # Skip custom initialization, uses torch by default 'uniform Xavier style' init
        return
    
    init_fn = INIT_MAP[weights_init]

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
