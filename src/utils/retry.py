"""Retry logic with exponential backoff."""

import time
import functools
from typing import Callable, Tuple, Type, Any, Optional

from ..core.logger import get_logger
from ..core import shutdown


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


# ─── Transient-error helpers (used by provider implementations) ────────────

_TRANSIENT_PATTERNS = (
    "timeout", "timed out",
    "network", "connection",
    "internal server error", "internal error",
    "bad gateway", "service unavailable", "gateway time-out",
    " 500", " 502", " 503", " 504",
    "500.", "502.", "503.", "504.",
    "deadline exceeded", "unavailable",
)


def is_transient_error(err: Exception) -> bool:
    """Best-effort classification of an exception as a retry-worthy transient.

    Pattern-matches the string form because both the Google SDK and ``requests``
    surface useful info in the message; the exception type is less reliable
    (the SDK wraps things under a generic ``Exception``).
    """
    msg = str(err).lower()
    return any(p in msg for p in _TRANSIENT_PATTERNS)


def interruptible_transient_retry(
    operation: Callable[[], Any],
    *,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    classifier: Callable[[Exception], bool] = is_transient_error,
    logger_name: Optional[str] = None,
) -> Any:
    """Call ``operation`` and retry only on classified-transient exceptions.

    Backoff is interruptible — SIGTERM during a sleep wakes immediately and
    causes the operation to raise (preserving the typed exception so callers
    can react). Non-transient exceptions propagate on the first occurrence.

    Args:
        operation: zero-arg callable performing the request.
        max_attempts: total attempts (including the first one).
        base_delay: initial backoff in seconds.
        max_delay: cap on a single backoff window.
        exponential_base: backoff multiplier.
        classifier: returns True when an exception should be retried.

    Returns:
        Whatever ``operation`` returns.
    """
    logger = get_logger(logger_name or __name__)
    last: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as e:
            last = e
            if attempt == max_attempts or not classifier(e):
                raise

            delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
            logger.warning(
                f"Transient error (attempt {attempt}/{max_attempts}): {e} — "
                f"retrying in {delay:.1f}s"
            )
            if not shutdown.interruptible_sleep(delay):
                raise  # shutdown requested — surface the in-flight error

    # Unreachable under normal control flow, kept to satisfy type-checkers.
    raise last if last else RuntimeError("interruptible_transient_retry: unreachable")


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
