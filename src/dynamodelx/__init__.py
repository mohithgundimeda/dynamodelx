from .tabular.UFA.ufa import UFA
from importlib.metadata import version, PackageNotFoundError
from .tabular.BNN.bnn_regressor import BnnRegressor

__all__ = ["UFA", "BnnRegressor"]

try:
    __version__ = version("dynamodelx")
except PackageNotFoundError:
    __version__ = "unknown"