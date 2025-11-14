import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Optional, Dict, Tuple
from ...utils.activations import validate_hidden_act, get_hidden_act, ActivationType
from ...utils.optimizer import validate_optimizer, get_optimizer, OptimizerType
from ...utils.weights_init import validate_weights_init, _init_weights, WeightInitType
from ...utils.device import validate_device, DeviceType
from ...utils.custom_arch import validate_custom_arch
from ...utils.loss import LossType, validate_loss
from ...utils.metrics import get_metrics
from ...utils.data import X_to_torch, preprocess_y, process_predictions, process_y_true

class UFA:

    ARCHITECTURE_MAP : Dict[str, Tuple]= {
        "small": (64, 32),
        "medium": (128, 64, 32),
        "large": (256, 128, 64, 32)
    }

    TASKS : Tuple = ('regression', 'classification')

    def __init__(self,
                 task : str,
                 model_size : Optional[str],
                 input_dim : int,
                 output_dim : int,
                 loss : LossType,

                 device: DeviceType = 'cuda',
                 weights_init : Optional[WeightInitType]= None,
                 hidden_activation : ActivationType = 'relu',
                 optimizer : OptimizerType = 'adam',
                 uncertainty : bool = False,
                 multiclass : bool = False,
                 custom_architecture : Optional[list[int]] = None,
                 return_metrics : bool = True,
                 auto_build : bool = True
                 ):
        
        self.task = task
        self.model_size = model_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss = loss
        self.device = device
        self.weights_init = weights_init
        self.hidden_activation = hidden_activation
        self.optimizer = optimizer
        self.uncertainty = uncertainty
        self.multiclass = multiclass
        self.custom_architecture = custom_architecture
        self.return_metrics = return_metrics
        self.auto_build = auto_build

        self.__post_init__()
    
    def _validate_basic(self) -> None:
        """
        Validates some basic requiements to proceed further
        """
        if self.task not in self.TASKS:
             raise ValueError(f"Invalid task '{self.task}'. Expected one of {self.TASKS}.")
        
        if (self.model_size not in self.ARCHITECTURE_MAP) and self.model_size is not None:
            raise ValueError(  f"Invalid model_size '{self.model_size}'. Expected one of {list(self.ARCHITECTURE_MAP.keys())} or None.")
        
        if not isinstance(self.input_dim, int):
            raise TypeError(f'Expected input_dim to be an integer, but recieved {type(self.input_dim)}')

        if not isinstance(self.output_dim, int):
            raise TypeError(f'Expected output_dim to be an integer, but recieved {type(self.output_dim)}')
        
        if not isinstance(self.uncertainty, bool):
            raise TypeError(f'Expected uncertainty to be either True or False')

        if not isinstance(self.multiclass, bool):
            raise TypeError(f'Expected multiclass to be either True or False')
        
        if not isinstance(self.return_metrics, bool):
            raise TypeError(f'Expected return_metrics to be either True or False')
    
    def _validate_model_ingredients(self) -> None:
        """
        Validates given specifications matches with model assumptions to build
        """
        self.device = validate_device(self.device)
        self.weights_init = validate_weights_init(self.weights_init)
        self.hidden_activation = validate_hidden_act(self.hidden_activation)
        self.optimizer = validate_optimizer(self.optimizer)
        self.custom_architecture =  validate_custom_arch(self.custom_architecture)
        self.loss, self.loss_function = validate_loss(self.loss)

        if self.return_metrics:
            self.metrics = get_metrics(self.task, self.multiclass)
    
    def _validate_architecture_logic(self) -> None:
        """
        Validates architecture logic for dynamic modeling
        """
        if not self.model_size and not self.custom_architecture:
            raise ValueError(
                "Atleast one of either model_size or custom_architecture should be specified, both can't be None"
            )

        if self.model_size and self.custom_architecture:
            raise ValueError(
                "Specify only one: either `model_size` or `custom_architecture`, not both."
            )

        if self.task == 'regression' and self.multiclass:
            raise ValueError(
                "Regression task cannot have multiclass=True."
            )

        if self.task == 'classification' and self.uncertainty:
            raise ValueError(
                "Uncertainity is a feature for regression tasks"
            )

        if self.task == 'classification' and not self.multiclass and self.output_dim > 1:
            raise ValueError(
                "For binary classification (multiclass=False), output_dim must be 1."
            )

        if (self.task == 'classification' and self.multiclass) and self.output_dim <= 1:
            raise ValueError(
                "For multiclass classification, output_dim must be greater than 1."
            )
        
        if (
            (self.task=='regression' and (self.loss == 'binary_cross_entropy' or self.loss == 'cross_entropy_loss'))
            or
            (self.task == 'classification' and (self.loss == 'mean_square_loss' or self.loss == 'gaussian_nll_loss'))
            ):
            raise ValueError(
                f'Loss Function mismatch, recieved {self.loss} for {self.task}'
            )
        
        if self.uncertainty and self.loss != 'gaussian_nll_loss':
            raise ValueError(
                f"Can't perform uncertainity without gaussian_nll_loss"
            )
        if not self.uncertainty and  self.loss == 'gaussian_nll_loss':
            raise ValueError(
                f"Suggesting to use mean_square_error instead of {self.loss}"
            )
        
        if self.multiclass and self.loss == 'binary_cross_entropy':
            raise ValueError(
                f"Suggesting to use cross_entropy_loss instead of {self.loss} for multiclass"
            )
        
        if not self.multiclass and self.loss == 'cross_entropy_loss':
            raise ValueError(
                f"Suggesting to use binary_cross_entropy instead of {self.loss} for binary classification"
            )

        
    def summary(self) -> None:
        """
        Print model's initialization summary after creating an instance
        """
        print("Model Configuration:")
        print(f"  Task:               {self.task}")
        print(f"  Model Size:         {self.model_size or 'Custom'}")
        print(f"  Input Dimension:    {self.input_dim}")
        print(f"  Output Dimension:   {self.output_dim}")
        print(f"  Loss                {self.loss}")
        print(f"  Device:             {self.device}")
        print(f"  Hidden Activation:  {self.hidden_activation}")
        print(f"  Optimizer:          {self.optimizer}")
        print(f"  Weights Init:       {self.weights_init or 'Default'}")
        print(f"  Uncertainty:        {self.uncertainty}")
        print(f"  Multiclass:         {self.multiclass}")
        print(f"  Custom Architecture:{self.custom_architecture}")

    def model_info(self) -> None:
        """
        Prints the built model structure and number of trainable parameters
        """
        try:
            model = self.model
        except AttributeError:
            raise ValueError("Error accessing model — it might not be initialized yet.")

        print("\n===== Built Model =====")
        print(model)
        print("=========================\n")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}\n")

        print("Model successfully built\n")


    def __post_init__(self):
        self._validate_basic()
        self._validate_model_ingredients()
        self._validate_architecture_logic()
        self.summary()

        if self.auto_build:
            print(f"Building the model ...")
            self.build()
            self.model_info()

    
    def __repr__(self):
        return (f"UFA(task={self.task!r}, model_size={self.model_size!r}, "
                f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"device={self.device!r}, uncertainty={self.uncertainty}, "
                f"loss={self.loss}, multiclass={self.multiclass})")
 
    def build(self) -> None:
        """
        Builds the dynamic model for given specification
        """
        architecture = self.ARCHITECTURE_MAP[self.model_size] if self.model_size else self.custom_architecture
        
        layers = []
        input_dim = self.input_dim
        for models in architecture:
            layers += [torch.nn.Linear(input_dim, models), get_hidden_act(self.hidden_activation)]
            input_dim = models

        layers += [torch.nn.Linear(input_dim, self.output_dim * 2 if self.uncertainty else self.output_dim)]
        
        self.model = torch.nn.Sequential(*layers).to(self.device)

        _init_weights(self.model, self.weights_init)

    def preprocess_user_data(self, X: np.ndarray, y: np.ndarray, val_size, test_size, batch_size) -> tuple[DataLoader, DataLoader, DataLoader]:
            
            X = X_to_torch(X, self.input_dim)
            y = preprocess_y(y)

            self.X_mean = X.mean(dim=0)
            self.X_std = X.std(dim=0)

            X = (X - self.X_mean) / self.X_std

            n_samples = X.shape[0]
            idx = torch.randperm(n_samples)

            train_val_split = int(n_samples * (1 - val_size))
            train_idx = idx[:train_val_split]
            eval_idx = idx[train_val_split:]

            val_test_split = int(len(eval_idx) * test_size)
            test_idx = eval_idx[:val_test_split]
            val_idx = eval_idx[val_test_split:]


            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory= self.device.type != 'cpu')
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory= self.device.type != 'cpu')
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory= self.device.type != 'cpu')

            return train_loader, val_loader, test_loader

    
    def train(
            self,
            X : np.ndarray,
            y:  np.ndarray,
            epochs : int,
            learning_rate : float,
            momentum : float | None = None,
            val_size : float = 0.2,
            test_size : float = 0.1,
            batch_size : int = 32
    ) -> dict:

        if not isinstance(epochs, int):
            raise TypeError(
                f'Expected epochs to be an integer'
            )
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                f"Expected learning_rate to be float"
            )
        if not isinstance(momentum, (int, float)):
            raise TypeError(
                f"Expected momentum to be float"
            )
        if not isinstance(val_size, float) or not (0 < val_size < 1):
            raise ValueError("Expected val_size to be a float between 0 and 1")

        if not isinstance(test_size, float) or not (0 < test_size < 1):
            raise ValueError("Expected test_size to be a float between 0 and 1")

        if not isinstance(batch_size, int):
            raise TypeError(
                f'Expected batch_size to be an integer'
            )
        
        
        train_loader, val_loader, test_loader = self.preprocess_user_data(
                                                                            X = X, 
                                                                            y = y, 
                                                                            val_size = val_size, 
                                                                            test_size = test_size, 
                                                                            batch_size = batch_size
                                                                            )

        optimizer_function = get_optimizer(self.optimizer, params= self.model.parameters(), lr = learning_rate, momentum = momentum if momentum else None)

        train_loss_track = []
        val_loss_track = []
        
        if self.return_metrics:
            metrics_tracking = {f'validation_{k}':[] for k in self.metrics.keys()}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_samples = 0
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(self.device, non_blocking=True), y_train_batch.to(self.device, non_blocking=True)
                num_samples = X_train_batch.shape[0]

                y_train_pred = self.model(X_train_batch)
                
                if self.task == "classification" and self.multiclass:
                    assert (
                        y_train_pred.shape[1] == self.output_dim and y_train_batch.ndim == 1
                    ), f"CrossEntropyLoss expects logits of shape (B, K) and targets (B,), got {y_train_pred.size()} vs {y_train_batch.size()}"

                elif self.task == "regression" and self.uncertainty:
                    assert (
                        y_train_pred.shape[1] == y_train_batch.shape[1] * 2
                    ), f"Uncertainty regression expects output_dim*2 neurons for mean+var, got {y_train_pred.shape} vs {y_train_batch.shape}"

                else:
                    assert(
                        y_train_pred.size() == y_train_batch.size()
                    ), f"Shape mismatch: y_pred={y_train_pred.size()} vs y_true={y_train_batch.size()}"
            
                loss_train = self.loss_function(y_train_pred, y_train_batch)

                optimizer_function.zero_grad()
                loss_train.backward()
                optimizer_function.step()


                train_loss += loss_train.item() * num_samples
                train_samples += num_samples
            
            avg_train_loss_per_sample = train_loss / train_samples
            train_loss_track.append(avg_train_loss_per_sample)

            self.model.eval()
            val_loss = 0
            val_samples = 0
            y_val_pred_concat = []
            y_val_true_concat = []
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    num_val_samples = X_val_batch.shape[0]
                    X_val_batch, y_val_batch = X_val_batch.to(self.device, non_blocking=True), y_val_batch.to(self.device, non_blocking=True)
                    y_val_pred = self.model(X_val_batch)
                    
                    loss_val = self.loss_function(y_val_pred, y_val_batch)

                    y_val_pred = process_predictions(
                                            y_pred = y_val_pred, 
                                            task=self.task, 
                                            multiclass = self.multiclass, 
                                            output_dim = self.output_dim,
                                            uncertainty = self.uncertainty
                                            )
                        
                    y_val_true = process_y_true(
                            y_true=y_val_batch,
                            task=self.task, 
                            multiclass = self.multiclass, 
                            output_dim = self.output_dim,
                            )
                    
                    y_val_pred_concat.append(y_val_pred.cpu().numpy())
                    y_val_true_concat.append(y_val_true.cpu().numpy())

                    val_loss += loss_val.item() * num_val_samples
                    val_samples += num_val_samples
                
            avg_val_loss_per_sample = val_loss/ val_samples
            val_loss_track.append(avg_val_loss_per_sample)

            y_val_pred_concat = np.concatenate(y_val_pred_concat, axis=0)
            y_val_true_concat = np.concatenate(y_val_true_concat, axis=0)

            assert y_val_pred_concat.shape == y_val_true_concat.shape, f"Mismatch validation shape, y_pred:{y_val_pred_concat.shape}, y_true:{y_val_true_concat.shape}"

            if self.return_metrics:
                for metric_name, func in self.metrics.items():
                    metrics_tracking[f'validation_{metric_name}'].append(func(y_val_true_concat, y_val_pred_concat))

            if (epoch+1) == epochs:
                print(
                    f"Final average train loss for a sample : {avg_train_loss_per_sample}\n",
                    f"Final average val loss for a sample : {avg_val_loss_per_sample}\n"
                )

        test_loss = 0
        test_samples = 0
        y_test_pred_concat = []
        y_test_true_concat = []

        self.model.eval()
        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:
                num_test_samples = X_test_batch.shape[0]

                X_test_batch, y_test_batch = (
                    X_test_batch.to(self.device, non_blocking=True),
                    y_test_batch.to(self.device, non_blocking=True),
                )

                y_test_pred = self.model(X_test_batch)
                loss_test = self.loss_function(y_test_pred, y_test_batch)

                y_test_pred = process_predictions(
                    y_pred=y_test_pred,
                    task=self.task,
                    multiclass=self.multiclass,
                    output_dim=self.output_dim,
                    uncertainty=self.uncertainty,
                )

                y_test_true = process_y_true(
                    y_true=y_test_batch,
                    task=self.task,
                    multiclass=self.multiclass,
                    output_dim=self.output_dim,
                )

                y_test_pred_concat.append(y_test_pred.cpu().numpy())
                y_test_true_concat.append(y_test_true.cpu().numpy())

                test_loss += loss_test.item() * num_test_samples
                test_samples += num_test_samples

        y_test_pred_concat = np.concatenate(y_test_pred_concat, axis=0)
        y_test_true_concat = np.concatenate(y_test_true_concat, axis=0)

        assert (
            y_test_pred_concat.shape == y_test_true_concat.shape
        ), f"Mismatch test shapes: pred={y_test_pred_concat.shape} true={y_test_true_concat.shape}"

        avg_test_loss = test_loss / test_samples

        if self.return_metrics:
            test_metrics = {
                f'test_{metric_name}': func(y_test_true_concat, y_test_pred_concat)
                for metric_name, func in self.metrics.items()
            }

        print(
            f"\nFinal Test Loss per sample: {avg_test_loss}\n"
        )

        
        if self.return_metrics:
            output_dict = {
                            'train': {'train_loss' : train_loss_track,},
                            'validation' : {'validation_loss' : val_loss_track, **metrics_tracking,},
                            'test' : {**test_metrics}
                        }
        else:
            output_dict = {
                            'train': {'train_loss' : train_loss_track},
                            'validation' : {'validation_loss' : val_loss_track}
                        }

        return output_dict

        

    def predict(self, input : np.ndarray) -> np.ndarray:
            """
            Generate predictions from trained model.

            Supports:
            1) Single-output regression
            2) Multi-output regression
            3) Single-output uncertainty regression
            4) Multi-output uncertainty regression
            5) Binary classification
            6) Multiclass classification
            """

            input = X_to_torch(input)

            try:
                input = (input - self.X_mean) / self.X_std
            except Exception:
                raise ValueError("Make sure train() is called before predict().")

            input = input.to(self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(input)

            output = process_predictions(
                y_pred=output,
                task=self.task,
                multiclass=self.multiclass,
                output_dim=self.output_dim,
                uncertainty=self.uncertainty
            )

            return output.cpu().numpy()


