# DynaModelX
___

## Introduction

__DynaModelX__ stands for Dynamic model extension, it is built as a side project for quick modeling with minimal UI. This package can dynamically build, train and return model performance according to the user given specifications.

DynaModelX contains __UFA__ (Universal Function Approximator), which can mainly perform 6 tasks on tabular data (atleast for now):

1. Univariate Regression
1. Multivariate Regression
1. Heteroscedastic Regression (Single Target)
1. Heteroscedastic Regression (Multi-Target)
1. Binary Classification (Bernoulli output)
1. Multi-Class Classification (Categorical Output)

__UFA__ is built on top of __Pytorch__, we can alternate with these tasks by just changing some couple of parameters.

## Installation

To install __DynaModelX__, run

```
pip install dynamodelx
```

To use __UFA__, import the estimator from dynamodelx

```python
from dynamodelx import UFA
```

## Quick Examples and Hyperparameters

For Univariate Regression,

```python
from sklearn.datasets import fetch_california_housing
from dynamodelx import UFA
from dynamodelx.plots import get_plots

data = fetch_california_housing()
X, y = data.data, data.target

ufa = UFA(  
            task = 'regression', 
            model_size = 'small', 
            input_dim = X.shape[1],
            output_dim = 1,
            loss = 'mean_square_loss',
            device = 'cuda',
            weights_init = 'he', 
            hidden_activation = 'relu',
            optimizer = 'adam',
            uncertainty = False,
            multiclass = False,
            custom_architecture = None, 
            return_metrics = True,
            auto_build = True
        )

performance = ufa.train(X, y, epochs=50, batch_size=32, learning_rate=0.01, val_size=0.3, test_size=0.2)

get_plots(performance=performance)

prediction = ufa.predict(X[:5])

ufa.save(path='california_housing.pth')

```

The hyperparameters for this estimator to initialize __UFA__ are as follows:

* **task** represents the kind of task we are performing, it takes a string. Either __'regression'__ or __'classification'__.
* **model_size** represents the size of the model, it takes one of these values:
    * __'small'__ - contains 2 hidden layers. 64 and 32 neurons respectively.
    * __'medium'__ - contains 3 hidden layers. 128, 64 and 32 neurons respectively.
    * __'large'__ - contains 4 layers. 256, 128, 64 and 32 neurons respectively.
    * __None__ - when we want to use our very own custom architecture, note the **model_size** should be __None__ if **custom_architecture** is given.(By default, **custom_architecture** is __None__. More about it discussed below)
* **input_dim** - takes an integer, the feature-space of data.
* **output_dim** - takes an integer, depends on the task we are solving. For example, 1 if it's univariate regression or heteroscedastic regression (single target) or binary classification, __d__ (required ouput dimension)if it's multivariate regression or heteroscedastic regression (multi-target) or  multi-class classification.
* **loss** represents the loss function we'll use to optimize the model, it takes one of these strings:
    * **'mean_square_loss'** -  a classic loss function for regression.
    * **'gaussian_nll_loss'** - for regression with uncertainty (heteroscedastic regression).
    * **'binary_cross_entropy'** - for binary classification.
    * **'cross_entropy_loss'** - for multi-class classification.
* **device** takes any of these strings:
    * **'cpu'** - model uses cpu to train.
    * **'cuda' or 'cuda:n'** - model uses cuda to train if it's available, where n is the gpu index.
    * By default, it's set to **'cuda'**.
* **weights_init** takes any of these values:
    * **'xavier'** - for xavier/glorot uniform weight initialization. Usually used with sigmoid/tanh activations. Acts with gain=1.0.
    * **'he'** - for he uniform weight initialization. Usually used with relu/leaky relu activations. Acts with a=0, mode='fan_in', nonlinearity='leaky_relu'.
    * **'uniform'** - for uniform weight initialization. Acts with a=0.0, b=1.0.
    * **'normal'** - for normal weight initialization. Acts with mean=0.0, std=1.0.
    * **None** - pytorch default weight initialization.
    * By default, it's set to **None**.
* **hidden_activation** takes any of these strings:
    * **'relu'** - Rectified Linear Unit activation.
    * **'leaky_relu'** - Leaky Rectified Linear Unit activation.
    * **'prelu'** - Parametric Rectified Linear Unit activation.
    * **'elu'** - Exponential Linear Unit activation.
    * **'sigmoid'** - Sigmoid activation.
    * **'tanh'** - Hyperbolic Tangent activation.
    * **'gelu'** - Gaussian Error Linear Unit activation.
    * **'mish'** - Mish activation.
    * By default, it's set to **'relu'**.
* **optimizer** takes any of these strings:
    * **'adam'** - Adam optimizer.
    * **'adamw'** - AdamW optimizer.
    * **'sgd'** - Stochastic Gradient Descent optimizer.
    * By default, it's set to **'adam'**.
* **uncertainty** takes a boolean value, True if we want to perform regression with uncertainty (heteroscedastic regression), False otherwise. By default, it's set to False. If it's set to True, the loss function should be __'gaussian_nll_loss'__ and the model will output both prediction and variance for each input sample. Note,
    * Making **uncertainty** True will work only when the **task** is set to __'regression'__.
    * Model also shows PICP and MPIW metrics for test data
    * For multi-target heteroscedastic regression, the model will output both prediction and variance for each target variable.
    * We don't have to include variance as a separate output dimension, the model will automatically takes care of it. If the **output_dim** is set to __d__, the model will output __d__ predictions and __d__ variance values.
* **multiclass** takes a boolean value, True if we want to perform multi-class classification, False otherwise. By default, it's set to False. If it's set to True, the **output_dim** should be greater than 1 and the loss function should be __'cross_entropy_loss'__. Note, making **multiclass** True will work only when the **task** is set to __'classification'__.
* **custom_architecture** takes a list of integers, representing the number of neurons in each hidden layer. For example, [128, 64, 32] represents a model with 3 hidden layers with 128, 64 and 32 neurons respectively. By default, it's set to __None__. If it's set to None, the model will be built according to the **model_size** parameter. If we want to use our own custom architecture, we need to set the **model_size** parameter to __None__ and provide the architecture through this parameter. Note this is only for hidden layers and the length of the list should be at least 1 and all the values should be positive integers greater than 0. The output layer neurons are defined by the **output_dim** parameter, no need to mention it here.
* **return_metrics** takes a boolean value, True if we want to return performance metrics after training, False otherwise. By default, it's set to True. If it's set to True, the __train__ method will return a dictionary of type **TrainingHistory** containing training loss, validation and test losses and their metrics.
* **auto_build** takes a boolean value, True if we want to automatically build the model after initialization, False otherwise. By default, it's set to True. If it's set to False, we need to manually call the __build__ methods after initialization to build the model.

The hyperparameters for the __train__ method of __UFA__ are as follows:

* __X__ and __y__ are the feature matrix and target vector respectively. Note that __y__ should be in integer format (0, 1, 2, ..., n-1) for multi-class classification or 0 and 1 for binary classification.
* **epochs** takes an integer, number of epochs to train the model.
* **learning_rate** takes a float value, learning rate for the optimizer.
* **momentum** takes a float value between 0 and 1, momentum for the SGD optimizer. By default, it's set to **None**. This parameter is used only when the **optimizer** is set to __'sgd'__.
* **val_size** takes a float value between 0 and 1, representing the proportion of validation data from the whole dataset. By default, it's set to 0.2.
* **test_size** takes a float value between 0 and 1, representing the proportion of test data from the whole dataset. By default, it's set to 0.1. 
* **batch_size** takes an integer, number of samples per batch. By default, it's set to 32.

__metrics__ returned for different tasks are as follows:
* For Regression:
    * Mean Absolute Error (MAE)
    * R2 Score

* For Classification:
    * Accuracy
    * Precision
    * Recall
    * F1 Score


## Plots
To get training and validation loss plots, use the function __get_plots__ from __dynamodelx.plots__

```python
from dynamodelx.plots import get_plots
get_plots(performance)
```

This function takes the performance dictionary of type **TrainingHistory** returned by the __train__ method of __UFA__ as input and plots the training, validation and test losses and their metrics.

## Saving The Model
To save the trained model, use the __save__ method of __UFA__

```python
ufa.save('model_name.pth')
```
This method takes the file name (with .pth, .pt, .ckpt, .bin extensions) as input and saves the trained model in the current working directory.

## Examples
More examples can be found in the [__examples__ ](https://github.com/mohithgundimeda/dynamodelx/tree/main/examples "examples") folder of the repository.

## License
This project is licensed under the MIT License - see the [__LICENSE__](https://github.com/mohithgundimeda/dynamodelx/blob/main/LICENSE "LICENSE") file for details.

## Acknowledgements
* Built on top of [__PyTorch__](https://pytorch.org/ "pytorch.org")
* Inspired by various machine learning libraries and frameworks.
