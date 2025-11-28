from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import torch
import numpy as np
from typing import Tuple

reg_metrics_map = {
    "mae": mean_absolute_error,
    "r2": r2_score,
}

def make_classification_metrics(multiclass: bool = False):
    """
    Returns metric functions with appropriate averaging for
    binary or multiclass classification.
    """
    avg = "macro" if multiclass else "binary"

    return {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(
            y_true, y_pred, average=avg, zero_division=0
        ),
        "recall": lambda y_true, y_pred: recall_score(
            y_true, y_pred, average=avg, zero_division=0
        ),
        "f1": lambda y_true, y_pred: f1_score(
            y_true, y_pred, average=avg, zero_division=0
        )
    }


def get_metrics(task : str, multiclass: bool) -> dict:
    """
    Returns valid metrics for the user given task
    """
    if not isinstance(task, str):
        raise ValueError(
            f"Error fetching metrics, expected task to be a string but recieved {type(task)}"
        )
    
    if task != 'regression':
        classification_metrics_map = make_classification_metrics(multiclass)
        return classification_metrics_map
    
    return reg_metrics_map


def PICP_MPIW(y_pred:np.ndarray, y_std:np.ndarray | torch.Tensor, y_true:np.ndarray) -> Tuple[list, list]:
    
    confidence_levels = torch.linspace(0.10, 0.90, 9)

    y_pred_t = torch.as_tensor(y_pred, dtype=torch.float32)
    y_std_t  = torch.as_tensor(y_std, dtype=torch.float32)
    y_true_t = torch.as_tensor(y_true, dtype=torch.float32)

    picp_list, mpiw_list = [], []
    
    for ci in confidence_levels:
        alpha = (1 + ci) / 2.0
        z_val = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * alpha - 1)

        lower = y_pred_t - z_val * y_std_t
        upper = y_pred_t + z_val * y_std_t
        
        if lower.ndim == 1:
            picp = ((y_true_t >= lower) & (y_true_t <= upper)).float().mean().item()
            mpiw = (upper - lower).mean().item()
        else:
            picp = ((y_true_t >= lower) & (y_true_t <= upper)).float().mean(dim=0).tolist()
            mpiw = (upper - lower).mean(dim=0).tolist()

        picp_list.append(picp)
        mpiw_list.append(mpiw)

    return picp_list, mpiw_list
