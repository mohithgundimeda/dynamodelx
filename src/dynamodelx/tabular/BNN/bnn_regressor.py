import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, Self
from dynamodelx.utils.device import DeviceType, validate_device
from dynamodelx.utils.activations import ActivationType, validate_hidden_act, get_hidden_act
from dynamodelx.utils.optimizer import OptimizerType, validate_optimizer, get_optimizer
from dynamodelx.utils.custom_arch import validate_custom_arch
from dynamodelx.utils.data import X_to_torch, preprocess_y
from dynamodelx.utils.metrics import get_metrics, PICP_MPIW
from ...utils.loss import N_ELBO_KLD, N_ELBO_NLL
from ...utils.TrainingHistory import TrainingHistory



class NN(nn.Module):
    def __init__(self, hidden_layers:list[int], input_dim:int, output_dim:int, activation:nn.Module, device: torch.device):
        super().__init__()
        
        self.weight_mu = nn.ParameterList()
        self.weight_log_var = nn.ParameterList()
        self.bias_mu = nn.ParameterList()
        self.bias_log_var = nn.ParameterList()

        
        self.activation = activation
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.__post_init__()
        self.to(device)
        
    def __post_init__(self):
        
        layers = [self.input_dim] + self.hidden_layers + [self.output_dim*2]
        
        for in_f, dim in zip(layers[:-1], layers[1:]):
            self.weight_mu.append(nn.Parameter(torch.randn(dim, in_f) * 0.05))  
            self.bias_mu.append(nn.Parameter(torch.randn(dim) * 0.05))

            self.weight_log_var.append(nn.Parameter(torch.full((dim, in_f), -3.0)))
            self.bias_log_var.append(nn.Parameter(torch.full((dim,), -3.0)))

    @staticmethod
    def logvar_to_std(log_var:torch.Tensor) -> torch.Tensor:
        return torch.exp(0.5 * log_var)
    
    def forward(self, X:torch.Tensor, sample:bool=True) -> torch.Tensor:
        
        for i in range(len(self.weight_mu)):
            w_mu = self.weight_mu[i]
            b_mu = self.bias_mu[i]
            
            w_logvar = self.weight_log_var[i]
            b_logvar = self.bias_log_var[i]
            
            if sample:
                w_epsilon = torch.randn_like(w_logvar)
                w = w_mu + self.logvar_to_std(w_logvar) * w_epsilon
                
                b_epsilon = torch.randn_like(b_logvar)
                b = b_mu +  self.logvar_to_std(b_logvar) * b_epsilon
            else:
                w , b = w_mu, b_mu
            
            X = torch.matmul(X, w.T) + b
            
            if i < (len(self.weight_mu)-1):
                X = self.activation(X)
        
        mean_split, logvar_split = torch.chunk(X, 2, dim=1)
        
        logvar_split = torch.clamp(logvar_split, min=-10, max=5) # model output always will in between -10 and 5, the gradients will be 0 for smaple that predicts outside clamp
        
        std = torch.exp(0.5 * logvar_split)
        
        std = torch.clamp(std, min=1e-6, max=50)
        
        return mean_split, std
                

class BnnRegressor:

    ARCHITECTURE_MAP : Dict[str, Tuple]= {
        "small": [64, 32],
        "medium": [128, 64, 32],
        "large": [256, 128, 64, 32]
    }
    
    def __init__(self,
                model_size : Optional[str],
                input_dim : int,
                output_dim : int,
                device: DeviceType = 'cuda',
                hidden_activation : ActivationType = 'relu',
                optimizer : OptimizerType = 'adam',
                mc_samples:int = 20,
                custom_architecture : Optional[list[int]] = None,
                return_metrics : bool = True,
                auto_build : bool = True
                ):
        """

        Args:
            model_size (Optional[str]): size of the model - ['small', 'medium', 'large']
            input_dim (int): input dimension
            output_dim (int): output dimension
            device (DeviceType, optional): Uses this device to train the model. Defaults to 'cuda'.
            hidden_activation (ActivationType, optional): Neural Net's activation function(same for all layers). Defaults to 'relu'.
            optimizer (OptimizerType, optional): Optimizer function to train the model. Defaults to 'adam'.
            mc_samples (int, optional): No of monte carlo samples to draw from the optimizing(at validation)/optimied(at inference) posterior. Defaults to 20.
            custom_architecture (Optional[list[int]], optional): List of integers who's length represents the no of layers and each integer represents the no of models in that layer. Defaults to None.
            return_metrics (bool, optional): Should the model return evaluated metrics ?. Defaults to True.
            auto_build (bool, optional): Builts the model automatically after initialization. Defaults to True.
        """
        self.model_size = model_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.hidden_activation = hidden_activation
        self.optimizer = optimizer
        self.custom_architecture = custom_architecture
        self.return_metrics = return_metrics
        self.auto_build = auto_build
        self.mc_samples = mc_samples
        
        self.__post_init__()

    def _validate_basic(self) -> None:
        """
        Validates some basic requiements to proceed further
        """
        
        if (self.model_size not in self.ARCHITECTURE_MAP) and self.model_size is not None:
            raise ValueError(  f"Invalid model_size '{self.model_size}'. Expected one of {list(self.ARCHITECTURE_MAP.keys())} or None.")
        
        if not isinstance(self.input_dim, int):
            raise TypeError(f'Expected input_dim to be an integer, but recieved {type(self.input_dim)}')

        if not isinstance(self.output_dim, int):
            raise TypeError(f'Expected output_dim to be an integer, but recieved {type(self.output_dim)}')
        
        if self.output_dim == 0:
            raise RuntimeError('output_dim should be atleast 1')
        
        if not isinstance(self.mc_samples, int):
            raise TypeError(f'Expected mc_samples to be an integer, but recieved {type(self.mc_samples)}')
        
        if not isinstance(self.return_metrics, bool):
            raise TypeError(f'Expected return_metrics to be either True or False')
        
        if not isinstance(self.auto_build, bool):
            raise TypeError(f'Expected auto_build to be either True or False')
    
    def _validate_model_ingredients(self) -> None:
        """
        Validates given specifications matches with model assumptions to build
        """
        self.device = validate_device(self.device)
        self.hidden_activation = validate_hidden_act(self.hidden_activation)
        self.optimizer = validate_optimizer(self.optimizer)
        self.custom_architecture =  validate_custom_arch(self.custom_architecture)
        
        if self.return_metrics:
            self.metrics = get_metrics(task='regression', multiclass=False)
    
    def _validate_architecture_logic(self) -> None:
        """
        Validates architecture logic for dynamic modeling
        """
        if not self.model_size and not self.custom_architecture:
            raise ValueError(
                "At least one of either model_size or custom_architecture should be specified, both can't be None"
            )

        if self.model_size and self.custom_architecture:
            raise ValueError(
                "Specify only one: either `model_size` or `custom_architecture`, not both."
            )

        
    def summary(self) -> None:
        """
        Print model's initialization summary after creating an instance
        """
        print("Model Configuration:\n")
        print(f"  Model Size:         {self.model_size or 'Custom'}")
        print(f"  Input Dimension:    {self.input_dim}")
        print(f"  Output Dimension:   {self.output_dim}")
        print(f"  Loss                Negative ELBO Loss")
        print(f"  Device:             {self.device}")
        print(f"  Hidden Activation:  {self.hidden_activation}")
        print(f"  Optimizer:          {self.optimizer}")
        print(f"  Custom Architecture:{self.custom_architecture}\n")

    def model_info(self) -> None:
        """
        Prints a clean, table summary of the BNN model.
        """
        try:
            model = self.model
        except AttributeError:
            raise RuntimeError(f"Model not found, error while building the model")

        print("\n====================================")
        print("        Model Summary         ")
        print("====================================")

        header = f"{'Layer (name)':<30} {'Shape':<20} {'Param #':<10}"
        print(header)
        print("-" * len(header))

        total_params = 0

        for name, param in model.named_parameters():
            shape = tuple(param.shape)
            params = param.numel()
            total_params += params

            print(f"{name:<30} {str(shape):<20} {params:<10}")

        print("-" * len(header))
        print(f"{'Total Parameters:':<30} {total_params:<20}")
        print("====================================\n")

    def __post_init__(self):
        self._validate_basic()
        self._validate_model_ingredients()
        self._validate_architecture_logic()
        self.summary()

        if self.auto_build:
            print("Building the model ...")
            self.build()
            self.model_info()

    def build(self) -> None:
        """
        Build the NN model for the user given specs
        """
        if not self.auto_build:
            print("Building the model ...")
        
        self.model = NN(hidden_layers=self.custom_architecture or self.ARCHITECTURE_MAP[self.model_size],
                        input_dim=self.input_dim,
                        output_dim=self.output_dim,
                        activation = get_hidden_act(self.hidden_activation),
                        device = self.device
                        )
        
        if not self.auto_build:
            self.model_info()
    
    def preprocess_user_data(self, X:np.ndarray, y:np.ndarray, val_size:float, test_size:float, batch_size:int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Preprocess user's data and return DataLoader instance for train, val, test datasets
        Args:
            X (np.ndarray):  Input independent data
            y (np.ndarray): Input dependent data
            val_size (float): Portion of data used for validation
            test_size (float): Portion of data used for test
            batch_size (int): No of samples in each batch

        Raises:
            ValueError: If number of samples in X dosen't matches with samples in y
            RuntimeError: If the dataset is empty
            RuntimeError: If there are no batches after splitting
        """
        
        X = X_to_torch(X, input_dim=self.input_dim)
        y = preprocess_y(y, task='regression', multiclass=False, output_dim=self.output_dim, uncertainty=False)
        
        if batch_size > X.shape[0]:
            raise RuntimeError(f'batch_size:{batch_size} is higher than the given samples X:{X.shape[0]}')
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) do not match.")
        
        num_samples = X.shape[0]
        
        self.X_mean = X.mean(dim=0)
        self.X_std = X.std(dim=0)

        self.X_std = torch.where(self.X_std == 0, torch.ones_like(self.X_std), self.X_std)

        X = (X - self.X_mean) / self.X_std
        
        self.y_mean = y.mean(dim = 0)
        self.y_std = y.std(dim = 0)
        self.y_std = torch.where(self.y_std == 0, torch.ones_like(self.y_std), self.y_std)
        
        y = (y - self.y_mean) / self.y_std
        
        idx = torch.randperm(num_samples)
        n_test = int(num_samples * test_size)
        n_val = int(num_samples * val_size)
        n_train = num_samples - n_val - n_test
        
        train_idx = idx[:n_train]
        val_idx = idx[n_train: n_train + n_val]
        test_idx = idx[n_train+n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        self.num_train_samples = X_train.shape[0]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory= self.device.type != 'cpu')
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory= self.device.type != 'cpu')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory= self.device.type != 'cpu')
        
        if len(train_loader) == 0:
            raise RuntimeError("Training data is empty. Adjust batch_size, val_size and test_size to ensure training data is available.")
        if len(val_loader) == 0:
            raise RuntimeError("Validation data is empty. Adjust batch_size, val_size and test_size to ensure validation data is available.")
        if len(test_loader) == 0:
            raise RuntimeError("Test data is empty. Adjust batch_size, val_size and test_size to ensure test data is available.")
        
        return train_loader, val_loader, test_loader
    
    def train(self, X:np.ndarray, y:np.ndarray, epochs:int, learning_rate:float, momentum:Optional[float] = None, val_size:float=0.2, test_size:float=0.1, batch_size:int=32, temp_lr:float=0.01, temp_epochs:int=100):
        """
        Train's the built model
        Args:
            X (np.ndarray): Input independent data
            y (np.ndarray): Input dependent data
            epochs (int): no of epochs the model should train for
            learning_rate (float): learning rate for optimiztion
            momentum (float): momentum for SGD optimizer
            val_size (float, optional): Portion of data used for validation, defaults to 0.2.
            test_size (float, optional): Portion of data used for test, defaults to 0.1.
            batch_size (int, optional): No of samples in each batch, defaults to 32.
            temp_lr (float, optional): Learning rate for temperature parameter.
            temp_epochs (int, optional): No of epochs for updating temperature parameter
        """
        if not hasattr(self, 'model'):
            raise RuntimeError(f'Call build() to build the model before training')
        
        if not isinstance(epochs, int):
            raise TypeError(
                f'Expected epochs to be an integer'
            )
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                f"Expected learning_rate to be float"
            )
        if not isinstance(momentum, (int, float, type(None))):
            raise TypeError(
                f"Expected momentum to be float or None"
            )
        if not isinstance(val_size, float) or not (0 < val_size < 1):
            raise ValueError("Expected val_size to be a float between 0 and 1")

        if not isinstance(test_size, float) or not (0 < test_size < 1):
            raise ValueError("Expected test_size to be a float between 0 and 1")
        
        if val_size + test_size >= 1.0:
            raise ValueError("Sum of val_size and test_size must be < 1.0")

        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        
        if not isinstance(temp_lr, (float, int)):
            raise ValueError(f'Expected temp_lr be a floating point or an integer, but recieved {type(temp_lr)}')
        if not isinstance(temp_epochs, int):
            raise ValueError(f'Expected temp_epochs to be an integer, but recieved {type(temp_epochs)}')
        
        for p in self.model.parameters():
            p.requires_grad = True

        
        train_loader, val_loader, test_loader = self.preprocess_user_data(X=X, y=y, val_size=val_size, test_size=test_size, batch_size=batch_size)
        
        kwargs = {
            'lr' : learning_rate,
            'params' : self.model.parameters()
        }

        if momentum is None:
            kwargs['momentum'] = None
        else:
            kwargs['momentum'] = momentum
        optimizer_function = get_optimizer(self.optimizer, **kwargs)
        
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
                
                y_train_mean, y_train_std = self.model(X = X_train_batch)
                
                loss_nll = N_ELBO_NLL(y_pred_mean=y_train_mean, y_pred_std=y_train_std, y_true=y_train_batch)
                
                kld = 0
                for i in range(len(self.model.weight_mu)):
                    kld += N_ELBO_KLD(mu=self.model.weight_mu[i], log_var=self.model.weight_log_var[i]) + N_ELBO_KLD(mu=self.model.bias_mu[i],  log_var=self.model.bias_log_var[i])
                
                try:
                    kld_scaled = (kld / self.num_train_samples)
                except AttributeError:
                    raise AttributeError(f"Didn't found the variable num_train_samples")
                
                loss = kld_scaled + loss_nll
                
                optimizer_function.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer_function.step()
                
                train_loss += loss * num_samples
                train_samples += num_samples
            
            avg_train_loss = train_loss / train_samples
            train_loss_track.append(avg_train_loss.detach().cpu().numpy())
            
            self.model.eval()
            val_loss = 0
            val_samples = 0
            
            if self.return_metrics:
                total_val_predictions = []
                total_val_true_values = []

            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = (
                        X_val_batch.to(self.device, non_blocking=True),
                        y_val_batch.to(self.device, non_blocking=True)
                    )
                    num_samples = X_val_batch.shape[0]

                    sampled_mean_pred = []
                    sampled_std_pred = []

                    for _ in range(self.mc_samples):
                        y_val_mean, y_val_std = self.model(X_val_batch)
                        sampled_mean_pred.append(y_val_mean)
                        sampled_std_pred.append(y_val_std)

                    mean_stack = torch.stack(sampled_mean_pred)
                    std_stack = torch.stack(sampled_std_pred)

                    y_val_mean_pred = mean_stack.mean(dim=0)

                    aleatoric_var = torch.clamp((std_stack**2).mean(dim=0), min=1e-6, max=1e3)
                    epistemic_var = torch.clamp(mean_stack.var(dim=0, unbiased=False), min=0, max=1e3)

                    total_var = aleatoric_var + epistemic_var
                    y_val_std_pred = torch.sqrt(torch.clamp(total_var, min=1e-6, max=1e3))
                    

                    loss_val = N_ELBO_NLL(
                        y_pred_mean=y_val_mean_pred,
                        y_pred_std=y_val_std_pred,
                        y_true=y_val_batch
                    )

                    val_loss += loss_val * num_samples
                    val_samples += num_samples

                    if self.return_metrics:
                        total_val_predictions.append(
                            y_val_mean_pred.detach().cpu().numpy()
                        )
                        total_val_true_values.append(
                            y_val_batch.detach().cpu().numpy()
                        )

            if self.return_metrics:
                total_val_predictions = np.concatenate(total_val_predictions, axis=0)
                total_val_true_values = np.concatenate(total_val_true_values, axis=0)

                assert total_val_predictions.shape[0] == val_samples
                assert total_val_true_values.shape == total_val_predictions.shape
                
                for k, fun in self.metrics.items():
                    metrics_tracking[f'validation_{k}'].append(
                        fun(total_val_true_values, total_val_predictions)
                    )

            
            avg_val_loss = val_loss / val_samples
            val_loss_track.append(avg_val_loss.detach().cpu().numpy())

            if (epoch+1) == epochs:
                print(
                    f"Average train loss per sample : {avg_train_loss:.2f}",
                    f"\nAverage validation loss per sample : {avg_val_loss:.2f}"
                )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        dataset = val_loader.dataset
        X_calib = dataset.tensors[0].to(self.device)
        y_calib = dataset.tensors[1].to(self.device)

        self.log_T = torch.nn.Parameter(torch.zeros(1, device=self.device))
        
        temp_opt= get_optimizer(optimizer = self.optimizer, lr=temp_lr, params=[self.log_T], momentum = momentum if momentum is not None else None)
        
        for _ in range(temp_epochs):

            mean_raw, std_raw = self.model(X_calib)
            mean_raw = mean_raw.detach()
            std_raw = std_raw.detach()

            T = torch.exp(self.log_T)

            var_scaled = (std_raw ** 2) * T
            std_scaled = torch.sqrt(var_scaled)

            loss = N_ELBO_NLL(mean_raw, std_scaled, y_calib)

            temp_opt.zero_grad()
            loss.backward()
            temp_opt.step()

        T = torch.exp(self.log_T.detach())
        
        test_loss = 0
        test_samples = 0 
        y_test_mean_pred_track = []
        y_test_std_pred_track = []
        y_test_true_track = []
        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:
                X_test_batch, y_test_batch = X_test_batch.to(self.device, non_blocking=True), y_test_batch.to(self.device, non_blocking=True)
                num_samples = X_test_batch.shape[0]
                
                y_test_mean_stack = []
                y_test_std_stack = []
        
                for _ in range(self.mc_samples):
                    y_test_mean_sample, y_test_std_sample = self.model(X_test_batch)
                    y_test_mean_stack.append(y_test_mean_sample)
                    y_test_std_stack.append(y_test_std_sample)
                
                y_test_mean_stack = torch.stack(y_test_mean_stack)
                y_test_std_stack  = torch.stack(y_test_std_stack) 

                epistemic_var = y_test_mean_stack.var(dim=0, unbiased=False)

                aleatoric_var = (y_test_std_stack**2).mean(dim=0) * T

                y_test_std_pred = torch.sqrt(aleatoric_var + epistemic_var)

                y_test_mean_pred = y_test_mean_stack.mean(dim=0)
                
                loss_test = N_ELBO_NLL(y_test_mean_pred, y_test_std_pred, y_test_batch)
                
                test_loss += loss_test * num_samples
                test_samples += num_samples
                
                y_test_mean_pred_track.append(y_test_mean_pred.detach().cpu().numpy())
                y_test_std_pred_track.append(y_test_std_pred.detach().cpu().numpy())
                y_test_true_track.append(y_test_batch.detach().cpu().numpy())
            
        print(f"Average Test Loss: {test_loss/test_samples:.2f}")
        
        y_test_mean_pred_track = np.concatenate(y_test_mean_pred_track, axis=0)
        y_test_std_pred_track = np.concatenate(y_test_std_pred_track, axis=0)
        y_test_true_track = np.concatenate(y_test_true_track, axis=0)

        if self.return_metrics:
            test_metrics = {
                f'test_{metric_name}': func(y_test_true_track, y_test_mean_pred_track)
                for metric_name, func in self.metrics.items()
    }

        
        picp_values, mpiw_values = PICP_MPIW(y_test_mean_pred_track, y_test_std_pred_track, y_test_true_track)

        print("\n" + "="*70)
        print("   PICP & MPIW across confidence levels on test data")
        print("="*70)

        CI = np.linspace(0.10, 0.90, 9)
        for i, cl in enumerate(CI):
            p = np.array(picp_values[i])
            m = np.array(mpiw_values[i])
            cl_pct = cl * 100

            if p.ndim == 0 or p.size == 1:
                print(f"{cl_pct:5.0f}% → PICP: {float(p):.4f} │ MPIW: {float(m):.4f}")
            else:
                ps = ", ".join(f"{x:.4f}" for x in p)
                ms = ", ".join(f"{x:.4f}" for x in m)
                print(f"{cl_pct:5.0f}% → PICP: [{ps}] │ MPIW: [{ms}]")
        
        print("="*70)
        
        print(f'Calibration Error : {np.mean(np.abs(np.array(picp_values) - CI)):.2f}')
        
        
        if self.return_metrics:
            output_dict = TrainingHistory(
                            train = {'train_loss' : train_loss_track,},
                            validation = {'validation_loss' : val_loss_track, **metrics_tracking,},
                            test = {**test_metrics}
                        )
        else:
            output_dict = TrainingHistory(
                            train = {'train_loss' : train_loss_track},
                            validation = {'validation_loss' : val_loss_track}
                        )

        
        return output_dict
    
    def predict(self, X:np.ndarray, mc_samples:int=20) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Takes input and predicts the ouput and it's standard-deviation
        Args:
        X (np.ndarray): Takes input samples to predict.
        mc_samples (int): Number of samples to sample from the posterior. Default to 20.
        """
        
        if not hasattr(self, 'model'):
            raise RuntimeError(f'Call build() to build the model before training')
        if mc_samples <= 0:
            raise RuntimeError('model needs atleast 1 mc_sample to predict.')
        
        
        X = X_to_torch(X, input_dim=self.input_dim)
        X = (X - self.X_mean) / self.X_std
        X = X.to(self.device)
        
        if not X.shape[0]:
            raise RuntimeError(f"Provided empty data")
        
        y_pred_stack = []
        y_std_stack = []
        
        try:
            self.model.eval()
        except AttributeError:
            raise AttributeError('No model is detected')
        
        with torch.no_grad():
            for _ in range(mc_samples):
                y_pred, y_std =self.model(X)
                y_pred_stack.append(y_pred)
                y_std_stack.append(y_std)
                
            y_pred_stack = torch.stack(y_pred_stack)
            y_std_stack = torch.stack(y_std_stack)
            
            aleatoric_var = (y_std_stack**2).mean(dim=0) * torch.exp(self.log_T)
            epistemic_var = y_pred_stack.var(dim=0, unbiased=False)
            
            y_pred_std = torch.sqrt(aleatoric_var + epistemic_var)
            
            y_pred = torch.mean(y_pred_stack, dim=0)
            
            y_pred = y_pred.detach().cpu().numpy() * self.y_std.numpy() + self.y_mean.numpy()
            
            y_pred_std = y_pred_std.detach().cpu().numpy() * self.y_std.numpy()
        
        return y_pred, y_pred_std
    
    def save(self, parameters_path:str, arguments_path:str) -> None:
        """
        Upon calling, the method will save the model's parameters and hyperparameters

        Args:
            parameters_path (str): Path to save the model parameter
            arguments_path (str): Path to save the model architecture
        """
        
        if not isinstance(parameters_path, str) or len(parameters_path.strip()) == 0:
            raise ValueError("Path must be a non-empty string.")

        if "." not in parameters_path:
            raise ValueError("Please provide a file name with extension, e.g., 'model_params.pt'")
        
        if not isinstance(arguments_path, str) or len(arguments_path.strip()) == 0:
            raise ValueError("Path must be a non-empty string.")

        if "." not in arguments_path:
            raise ValueError("Please provide a file name with extension, e.g., 'model_args.pt'")

        state_ext = parameters_path.split(".")[-1].lower()
        arch_ext = arguments_path.split(".")[-1].lower()

        allowed_exts = {"pth", "pt", "ckpt", "bin"}
        
        if state_ext not in allowed_exts:
            print(f"Warning: parameters_path extension '.{state_ext}' is unusual for PyTorch models. "
                f"Recommended: .pth, .pt, .ckpt")
        
        if arch_ext not in allowed_exts:
            print(f"Warning: arguments_path extension '.{arch_ext}' is unusual for PyTorch models. "
                f"Recommended: .pth, .pt, .ckpt")

        if not hasattr(self, "model"):
            raise RuntimeError("Model not built yet. Call build() or train() before saving.")
        
        model_args = {
            "model_size" : self.model_size,
            "input_dim" : self.input_dim,
            "output_dim" : self.output_dim,
            "device": self.device.type,
            "hidden_activation" : self.hidden_activation,
            "optimizer" : self.optimizer,
            "mc_samples": self.mc_samples,
            "custom_architecture" : self.custom_architecture,
            "return_metrics" : self.return_metrics,
            "auto_build" : self.auto_build,
            "X_mean" : self.X_mean,
            "X_std" : self.X_std,
            "y_mean" : self.y_mean,
            "y_std" : self.y_std,
            "log_T" : self.log_T.detach().cpu()
        }
        
        try:
            torch.save(self.model.state_dict(), parameters_path)
            torch.save(model_args, arguments_path)
        except Exception as e:
            raise RuntimeError(f"Error saving model: {e}")
         
        print(f"Model's state successfully saved to: {parameters_path}")
        print(f"Model's architecture successfully saved to: {arguments_path}")
    
    @classmethod
    def load(cls, parameters_path:str, arguments_path:str) -> Self:
        """
        Loading the saved architecture and model's trained parameters
        """
        
        args = torch.load(arguments_path, map_location='cpu')
        bnn = cls(
            model_size=args.get('model_size'),
            input_dim=args.get('input_dim'),
            output_dim=args.get('output_dim'),
            device=args.get('device'),
            hidden_activation=args.get('hidden_activation'),
            optimizer=args.get('optimizer'),
            mc_samples=args.get('mc_samples'),
            custom_architecture=args.get('custom_architecture'),
            return_metrics=args.get('return_metrics'),
            auto_build=True
        )
        
        
        state_dict = torch.load(parameters_path, map_location=args['device'])
        bnn.model.load_state_dict(state_dict)
        
        bnn.X_mean = args['X_mean']
        bnn.X_std = args['X_std']
        bnn.y_mean = args['y_mean']
        bnn.y_std = args['y_std']
        bnn.log_T = args['log_T'].to(device=args['device'])
        return bnn
        
        
    
        
        
        
        
        
        
            
            
            
            

            
                
                
                                
        
        
        
        


