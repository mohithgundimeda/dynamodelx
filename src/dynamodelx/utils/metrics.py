from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

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
        ),
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