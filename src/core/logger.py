"""Logging infrastructure with file and console handlers."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Global logger cache
_loggers = {}


def get_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Get or create a logger with console and file handlers.

    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler with detailed format
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _loggers[name] = logger
    return logger


def setup_root_logger(log_file: str = "logs/synthesizer.log", level: str = "INFO"):
    """Setup the root logger for the application.

    Args:
        log_file: Path to log file
        level: Logging level
    """
    return get_logger("data_synthesizer", log_file, level)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_execution_time(func):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = datetime.now()
        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start).total_seconds()
            logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    return wrapper


if __name__ == "__main__":
    print("=" * 60)
    print("LOGGER MODULE TEST")
    print("=" * 60)

    # Test 1: Basic logger creation
    logger = get_logger("test_module")
    logger.info("Test message - INFO level")
    logger.debug("Test message - DEBUG level (should not show in console)")
    print("  ✓ Basic logger created and tested")

    # Test 2: Logger with file output
    logger_with_file = get_logger("test_file", "logs/test.log", "DEBUG")
    logger_with_file.info("File logger test")
    logger_with_file.debug("Debug message to file")
    print("  ✓ File logger created")

    # Test 3: Logger caching
    logger2 = get_logger("test_module")
    assert logger is logger2, "Logger should be cached"
    print("  ✓ Logger caching works")

    # Test 4: LoggerMixin
    class TestClass(LoggerMixin):
        def do_something(self):
            self.logger.info("TestClass doing something")

    obj = TestClass()
    obj.do_something()
    print("  ✓ LoggerMixin works")

    # Test 5: Execution time decorator
    @log_execution_time
    def slow_function():
        import time
        time.sleep(0.1)
        return "done"

    result = slow_function()
    print(f"  ✓ Execution time decorator works: {result}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
