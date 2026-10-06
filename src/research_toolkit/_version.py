"""Installed distribution version, resolved once without importing optional dependencies."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("research-toolkit")
except PackageNotFoundError:  # Imported from a source tree that was never installed.
    __version__ = "uninstalled"
