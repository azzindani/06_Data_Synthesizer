"""Core infrastructure components."""

def __getattr__(name):
    if name == 'Config':
        from .config import Config
        return Config
    if name == 'load_config':
        from .config import load_config
        return load_config
    if name == 'get_logger':
        from .logger import get_logger
        return get_logger
    if name == 'ProgressManager':
        from .progress import ProgressManager
        return ProgressManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['Config', 'load_config', 'get_logger', 'ProgressManager']
