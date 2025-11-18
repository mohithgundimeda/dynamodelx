import torch
from typing import Literal, Dict, Type

HIDDEN_ACT_MAP: Dict[str, Type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "leakyrelu": torch.nn.LeakyReLU,
    "prelu": torch.nn.PReLU,
    "elu": torch.nn.ELU,
    "tanh": torch.nn.Tanh,
    "gelu": torch.nn.GELU,
    "mish": torch.nn.Mish,
}

ActivationType = Literal[
    "relu",
    "leakyrelu",
    "prelu",
    "elu",
    "tanh",
    "gelu",
    "mish",
]

def validate_hidden_act(activation: str) -> str:
    """
    Validates the user prefered activation function
    Raises ValueError and TypeError if invalid.
    """
    if not isinstance(activation, str):
        raise TypeError(f"Expected a string, but received {type(activation)}")

    activation = activation.lower().strip()

    if activation not in HIDDEN_ACT_MAP:
        raise ValueError(
            f"Invalid activation '{activation}'. Choose from: {list(HIDDEN_ACT_MAP.keys())}"
        )
    return activation
    

def get_hidden_act(activation: str) -> torch.nn.Module:
    """
    Returns an activation module instance for the given activation name.
    """  

    activation_class = HIDDEN_ACT_MAP[activation]
    return activation_class()
