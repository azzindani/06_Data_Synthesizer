"""Process-wide shutdown signal — interruptible across modules.

Anywhere in the codebase can read ``is_requested()`` to check whether a
SIGTERM/SIGINT has arrived, or call ``interruptible_sleep(seconds)`` to
wait on a long delay that wakes up immediately when shutdown is signaled.

The synthesizer installs the signal handlers via ``install_handlers()`` from
its base class. Providers (e.g. Gemini's daily-quota wait) just use
``interruptible_sleep`` without needing a callback reference.
"""

import logging
import signal
import threading
from typing import Optional

_event = threading.Event()
_logger = logging.getLogger(__name__)
_handlers_installed = False


def install_handlers() -> None:
    """Install SIGTERM/SIGINT handlers that set the shutdown event.

    Idempotent — safe to call from multiple places. Tests can call
    ``reset()`` between runs.
    """
    global _handlers_installed
    if _handlers_installed:
        return

    def _handle(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        _logger.info("Received %s — requesting shutdown", sig_name)
        _event.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    _handlers_installed = True


def request() -> None:
    """Programmatically request shutdown (used in tests / target-reached)."""
    _event.set()


def is_requested() -> bool:
    """Return True if shutdown has been signaled."""
    return _event.is_set()


def reset() -> None:
    """Clear the shutdown flag (tests only)."""
    _event.clear()


def interruptible_sleep(seconds: float, log_every: Optional[float] = None,
                        log_message_fn=None) -> bool:
    """Sleep for ``seconds`` but wake immediately if shutdown is requested.

    Args:
        seconds: total time to wait (negative/zero returns immediately).
        log_every: optional period between heartbeat log lines.
        log_message_fn: callable taking ``remaining_seconds`` and returning
            the message to log. Only invoked when ``log_every`` is set.

    Returns:
        True if the full duration elapsed; False if shutdown was requested.
    """
    if seconds <= 0:
        return not _event.is_set()

    remaining = float(seconds)
    chunk = min(remaining, log_every) if log_every else remaining

    while remaining > 0:
        # Event.wait returns True when set, False on timeout.
        if _event.wait(timeout=min(chunk, remaining)):
            return False
        remaining -= chunk
        if log_every and remaining > 0 and log_message_fn is not None:
            try:
                log_message_fn(remaining)
            except Exception:
                pass  # logging shouldn't break the sleep loop
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("SHUTDOWN MODULE TEST")
    print("=" * 60)

    import time

    # Test 1: no shutdown, sleep completes
    reset()
    start = time.monotonic()
    completed = interruptible_sleep(0.2)
    elapsed = time.monotonic() - start
    assert completed and 0.15 < elapsed < 0.5, f"unexpected elapsed={elapsed}"
    print(f"  ✓ sleep completes when no shutdown ({elapsed:.2f}s)")

    # Test 2: request() interrupts an ongoing sleep
    reset()
    timer = threading.Timer(0.1, request)
    timer.start()
    start = time.monotonic()
    completed = interruptible_sleep(5.0)
    elapsed = time.monotonic() - start
    assert not completed and elapsed < 1.0, f"unexpected elapsed={elapsed}"
    print(f"  ✓ sleep interrupted by request() ({elapsed:.2f}s)")

    # Test 3: install_handlers is idempotent
    reset()
    install_handlers()
    install_handlers()
    print("  ✓ install_handlers is idempotent")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
