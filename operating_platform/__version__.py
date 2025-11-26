"""To enable `operating_platform.__version__`"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("operating_platform")
except PackageNotFoundError:
    __version__ = "unknown"
