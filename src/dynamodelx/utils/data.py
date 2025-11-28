import numpy as np
import torch

def X_to_torch(X: np.ndarray | list | torch.Tensor | tuple, input_dim : int) -> torch.Tensor:
    """
    Converts input to torch.float32 tensor.
    Accepts only Tensor, NumPy array, or list.
    """
    if not isinstance(X, (torch.Tensor, np.ndarray, list, tuple)):
        raise TypeError("Input must be a torch.Tensor, numpy.ndarray, or list/tuple.")
    
    if isinstance(X, torch.Tensor):
        if not torch.is_floating_point(X):
            X = X.float()
    else:
        X = torch.as_tensor(X, dtype=torch.float32)

    if X.ndim > 2:
        raise  ValueError(f"Expected X to have at most 2 dimensions, but got {X.ndim}.")

    if X.ndim == 1 :
        X = X.reshape(-1,1)
    
    if X.shape[0] == 0:
        raise RuntimeError("Empty dataset provided.")
    
    if X.shape[1] != input_dim:
        raise RuntimeError(
            f"Model expects input dimension of {input_dim} but recieved {X.shape[1]}"
        )
    
    return X

def preprocess_y(
            y,
            task: str,
            multiclass: bool,
            output_dim: int,
            uncertainty: bool,
        ) -> torch.Tensor:
            """
            Preprocess target (y) values for regression and classification tasks.

            Handles:
            - Regression (single / multi-output, with or without uncertainty)
            - Binary and multiclass classification
            - Converts to correct dtype and shape expected by PyTorch losses

            Returns
            -------
            torch.Tensor:
                Properly shaped and typed tensor for model training.
            """
            if not isinstance(y, (torch.Tensor, np.ndarray, list, tuple)):
                raise TypeError("Input must be a torch.Tensor, numpy.ndarray, or list/tuple.")
            
            if not isinstance(y, (np.ndarray, torch.Tensor)):
                y = np.array(y)
            
            if y.shape[0] == 0:
                raise RuntimeError(f'Given y is empty')

            if task == "classification":
                if not multiclass:
                    unique = np.unique(y)
                    if not set(unique).issubset({0, 1}):
                        raise ValueError(f"Binary classification expects labels {{0, 1}}, got {unique.tolist()}")

                    if y.ndim == 1:
                        y = y.reshape(-1, 1)
                    elif y.ndim != 2 or y.shape[1] != 1:
                        raise ValueError(f"Binary y must be 1D or (N,1), got shape {y.shape}")

                    y = torch.as_tensor(y, dtype=torch.float32)

                else:
                    unique = np.unique(y)
                    expected = list(range(output_dim))
                    if not np.array_equal(unique, expected):
                        raise ValueError(
                            f"Multiclass needs labels {expected}, got {unique.tolist()}"
                        )

                    if isinstance(y, np.ndarray):
                        if not np.issubdtype(y.dtype, np.integer):
                            raise ValueError(f"Multiclass labels must be integers, got {y.dtype}")
                    else:
                        if y.dtype not in (torch.int16, torch.int32, torch.int64):
                            raise ValueError(f"Multiclass labels must be int tensor, got {y.dtype}")

                    if y.ndim != 1:
                        raise ValueError(f"Multiclass y must be 1D, got shape {y.shape}")

                    y = torch.as_tensor(y, dtype=torch.long)

            elif task == "regression":
                if uncertainty:
                    if y.ndim == 1:
                        if output_dim != 1:
                            raise RuntimeError(f'Expecting y to be a {output_dim} array/tensor. Suggesting to remove standard-deviation dimension from output_dim if it is included. As model will handle it automatically')
                        y = np.expand_dims(y, axis=1)
                    elif y.ndim != 2:
                        raise ValueError(
                            f"For uncertainty regression, expected y shape (N, {output_dim}), got {y.shape}."
                        )
                    elif y.shape[1] != output_dim:
                        raise ValueError(
                            f"For uncertainty regression, expected y shape (N, {output_dim}), got (N, {y.shape[1]}). Suggesting to remove standard-deviation dimension from output_dim if it is included. As model will handle it automatically"
                        )
                    y = torch.as_tensor(y, dtype=torch.float32)

                else:
                    if output_dim == 1:
                        if y.ndim == 1:
                            y = np.expand_dims(y, axis=1)
                        elif y.ndim != 2:
                            raise RuntimeError(f"Expected y to be 1D or 2D, got {y.shape}.")
                        elif y.shape[1] != 1:
                            raise RuntimeError(f'Expected y to have single dimension, but recieved {y.shape[1]}')
                    else:
                        if y.ndim == 1:
                            raise RuntimeError(
                                f"Expected y shape (N, {output_dim}) for multi-output regression, got (N,)."
                            )
                        elif y.ndim != 2:
                            raise RuntimeError(f"Expected y to be 2D, got {y.shape}.")
                        elif y.shape[1] != output_dim:
                            raise RuntimeError(f'Expecting y to be a {output_dim} dimension array/tensor, but recieved {y.shape[1]}')
                        
                    y = torch.as_tensor(y, dtype=torch.float32)

            else:
                raise ValueError(f"Unknown task type '{task}'. Expected 'regression' or 'classification'.")

            return y

def process_predictions(y_pred: torch.Tensor,
                        task: str,
                        multiclass: bool,
                        output_dim: int,
                        uncertainty: bool) -> torch.Tensor:
    """
    Standardizes model predictions for metrics/evaluation.

    Handles:
    - Regression (single & multi-output)
    - Uncertainty regression (single & multi-output)
    - Binary classification
    - Multiclass classification
    """
    if task == "regression" and uncertainty:
        y_pred , _ = torch.chunk(y_pred, 2, dim=1)
    
    # Applies to:
    #   - single-output regression
    #   - single-output uncertainty regression
    #   - binary classification logits (B,1)
    if output_dim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(dim=1) 

    if task == "classification":

        if multiclass:
            probs = torch.softmax(y_pred, dim=1)  
            y_pred = torch.argmax(probs, dim=1)   

        else:
            probs = torch.sigmoid(y_pred)
            y_pred = probs.round()

    return y_pred

def process_y_true(y_true: torch.Tensor, task: str, multiclass: bool, output_dim: int) -> torch.Tensor:
    """
    Standardizes y_true to the correct shape.
    Handles:
    - regression (B,)
    - binary classification (B,) or (B,1)
    - multiclass (B,) class indices
    """
    
    if task == "regression":
        return y_true.squeeze(-1) if y_true.ndim == 2 and y_true.shape[1] == 1 else y_true

    if task == "classification":
        
        if not multiclass:
            if output_dim == 1:
                return y_true.squeeze(-1) if y_true.ndim == 2 else y_true
        
        return y_true.squeeze(-1) if y_true.ndim == 2 else y_true

    return y_true
