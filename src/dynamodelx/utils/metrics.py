from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import torch

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


def PICP_MPIW(y_pred, y_var, y_true):
    confidence_levels = torch.linspace(0.10, 0.90, 9)

    y_pred_t = torch.from_numpy(y_pred).float().reshape(-1)
    y_var_t  = torch.from_numpy(y_var).float().reshape(-1)
    y_true_t = torch.from_numpy(y_true).float().reshape(-1)

    std_t = torch.sqrt(torch.nn.functional.softplus(y_var_t).clamp(min=1e-6))

    picp_list, mpiw_list = [], []

    sqrt2 = torch.sqrt(torch.tensor(2.0))

    for p in confidence_levels:
        p = torch.clamp(p, 1e-6, 1 - 1e-6)

        z_val = torch.abs(sqrt2 * torch.erfinv(2 * p - 1))

        lower = y_pred_t - z_val * std_t
        upper = y_pred_t + z_val * std_t

        picp = ((y_true_t >= lower) & (y_true_t <= upper)).float().mean().item()
        mpiw = (upper - lower).mean().item()

        picp_list.append(picp)
        mpiw_list.append(mpiw)

    return picp_list, mpiw_list
