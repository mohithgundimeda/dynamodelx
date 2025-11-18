import torch
from typing import Literal

DeviceType = Literal['cpu', 'cuda']


def validate_device(device : str) -> torch.device:
    """
    Validates the device type and it's compatability with system architecture
    """

    if not isinstance(device, str):
        raise TypeError(f'Invalid type, expected device to be a string but recieved {type(device)}')

    device = device.lower().strip()

    if device == 'cpu':
        return torch.device('cpu')
    
    if not torch.cuda.is_available():
        raise ValueError(f'CUDA is not available on this system')
    
    if device == 'cuda':
        return torch.device('cuda')
    
    if device.startswith("cuda:"):
        try:
            index = int(device.split(":")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid CUDA device format: '{device}'")

        if index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device '{device}' not found. Available devices: 0 to {torch.cuda.device_count() - 1}"
            )
        
        return torch.device(device)
    
    raise ValueError(f"Invalid device string '{device}'. Use 'cpu', 'cuda', or 'cuda:n'")