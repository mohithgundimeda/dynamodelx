from .tabular.UFA.ufa import UFA
from importlib.metadata import version, PackageNotFoundError

__all__ = ["UFA"]

try:
    __version__ = version("dynamodelx")
except PackageNotFoundError:
    __version__ = "unknown"