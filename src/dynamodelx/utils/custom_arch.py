import torch
from typing import Optional

def validate_custom_arch(arch: Optional[list[int]]) -> Optional[list[int]]:
    """
    Validates the custom_architecture, raises errors if invalid.
    Accepts None or a non-empty list of integers.
    """
    if arch is None:
        return None
    
    if not (isinstance(arch, list) and all(isinstance(i, int) for i in arch)):
        raise TypeError("Expected custom_architecture to be a list[int] or None.")
    
    if len(arch) == 0:
        raise ValueError("custom_architecture cannot be an empty list. Provide at least one layer.")
    
    return arch
