"""Retry logic with exponential backoff."""

import time
import functools
from typing import Callable, Tuple, Type, Any

from ..core.logger import get_logger


def retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential calculation
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


def retry_operation(
    operation: Callable,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable = None
) -> Any:
    """Retry an operation with exponential backoff.

    Args:
        operation: Callable to execute
        max_attempts: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Base for exponential calculation
        exceptions: Tuple of exceptions to catch
        on_retry: Optional callback on retry

    Returns:
        Result from successful operation

    Raises:
        Last exception if all attempts fail
    """
    logger = get_logger(__name__)
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()

        except exceptions as e:
            last_exception = e

            if attempt == max_attempts:
                logger.error(f"Operation failed after {max_attempts} attempts: {e}")
                raise

            delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)

            logger.warning(
                f"Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay:.1f}s"
            )

            if on_retry:
                on_retry(attempt, e, delay)

            time.sleep(delay)

    raise last_exception


if __name__ == "__main__":
    print("=" * 60)
    print("RETRY MODULE TEST")
    print("=" * 60)

    # Test 1: Decorator with eventual success
    attempt_count = 0

    @retry_with_backoff(max_attempts=3, base_delay=0.1, exceptions=(ValueError,))
    def eventually_succeeds():
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError(f"Attempt {attempt_count} failed")
        return "success"

    result = eventually_succeeds()
    print(f"  ✓ Decorator test: {result} after {attempt_count} attempts")

    # Test 2: Decorator with all failures
    @retry_with_backoff(max_attempts=2, base_delay=0.1, exceptions=(RuntimeError,))
    def always_fails():
        raise RuntimeError("Always fails")

    try:
        always_fails()
        print("  ✗ Should have raised exception")
    except RuntimeError:
        print("  ✓ Exception raised after max attempts")

    # Test 3: retry_operation function
    op_attempts = 0

    def retry_op():
        global op_attempts
        op_attempts += 1
        if op_attempts < 2:
            raise IOError("Network error")
        return "connected"

    result = retry_operation(
        retry_op,
        max_attempts=3,
        base_delay=0.1,
        exceptions=(IOError,)
    )
    print(f"  ✓ retry_operation: {result}")

    # Test 4: With on_retry callback
    retry_events = []

    def on_retry_callback(attempt, error, delay):
        retry_events.append(attempt)

    cb_attempts = 0

    def op_with_callback():
        global cb_attempts
        cb_attempts += 1
        if cb_attempts < 3:
            raise Exception("Fail")
        return "done"

    result = retry_operation(
        op_with_callback,
        max_attempts=3,
        base_delay=0.1,
        on_retry=on_retry_callback
    )
    print(f"  ✓ Callback test: {len(retry_events)} retries logged")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
