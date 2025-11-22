"""Utility functions and helpers."""

from .retry import retry_with_backoff
from .json_parser import extract_json, parse_json_safely

# Lazy imports for heavier modules
def __getattr__(name):
    if name == 'CostTracker':
        from .cost_tracker import CostTracker
        return CostTracker
    if name == 'NotificationManager':
        from .notifications import NotificationManager
        return NotificationManager
    if name == 'LocalOutputHandler':
        from .output_handler import LocalOutputHandler
        return LocalOutputHandler
    if name == 'OutputManager':
        from .output_handler import OutputManager
        return OutputManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'retry_with_backoff', 'extract_json', 'parse_json_safely',
    'CostTracker', 'NotificationManager', 'LocalOutputHandler', 'OutputManager'
]
