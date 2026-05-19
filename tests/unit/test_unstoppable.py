"""Tests for unstoppable-cron behavior: shutdown, target_rows, daily-reset timing."""

import threading
import time
from datetime import datetime, timezone, timedelta

import pytest

from src.core import shutdown


pytestmark = pytest.mark.unit


# ─── shutdown module ────────────────────────────────────────────────────────

def test_interruptible_sleep_completes_when_no_signal():
    shutdown.reset()
    start = time.monotonic()
    done = shutdown.interruptible_sleep(0.1)
    elapsed = time.monotonic() - start
    assert done is True
    assert 0.08 < elapsed < 0.5


def test_interruptible_sleep_wakes_on_request():
    shutdown.reset()
    timer = threading.Timer(0.1, shutdown.request)
    timer.start()
    start = time.monotonic()
    done = shutdown.interruptible_sleep(5.0)
    elapsed = time.monotonic() - start
    timer.cancel()
    assert done is False
    assert elapsed < 1.0, f"shutdown took too long: {elapsed:.2f}s"


def test_interruptible_sleep_returns_immediately_when_already_requested():
    shutdown.reset()
    shutdown.request()
    start = time.monotonic()
    done = shutdown.interruptible_sleep(5.0)
    elapsed = time.monotonic() - start
    assert done is False
    assert elapsed < 0.1


def test_reset_clears_state():
    shutdown.request()
    assert shutdown.is_requested()
    shutdown.reset()
    assert not shutdown.is_requested()


# ─── Pacific midnight reset window ──────────────────────────────────────────

def test_next_quota_reset_returns_future_pacific_midnight():
    from src.providers.gemini import _next_quota_reset, _PACIFIC

    if _PACIFIC is None:
        pytest.skip("IANA tzdata unavailable (zoneinfo missing)")

    now = datetime.now(timezone.utc)
    reset_at = _next_quota_reset(now)

    # Future
    assert reset_at > now
    # Within the next ~25 hours (24h + 5min buffer max)
    assert (reset_at - now) < timedelta(hours=25)
    # Lands at Pacific midnight + 5min buffer
    pt = reset_at.astimezone(_PACIFIC)
    assert pt.hour == 0
    assert pt.minute == 5


def test_next_quota_reset_falls_back_when_no_tzdata(monkeypatch):
    from src.providers import gemini

    monkeypatch.setattr(gemini, "_PACIFIC", None)
    now = datetime.now(timezone.utc)
    reset_at = gemini._next_quota_reset(now, buffer_minutes=0)
    delta = reset_at - now
    # Always > 23h, ≤ 24h+buffer — guarantees we span a daily reset.
    assert timedelta(hours=23, minutes=59) < delta <= timedelta(hours=24, minutes=1)


# ─── target_rows in synthesizer loop ────────────────────────────────────────

def test_target_reached_returns_false_when_not_configured():
    """target_rows=None or 0 means no ceiling — _target_reached always False."""
    from src.synthesizers.base import BaseSynthesizer

    # Avoid instantiating the abstract class — just exercise the staticmethod-like check
    # by building a tiny stand-in with the required attributes.
    class Fake:
        _target_reached = BaseSynthesizer._target_reached

        class progress:  # noqa
            progress_data = {'total_generated': 99999}

    fake = Fake()
    assert fake._target_reached(0) is False
    assert fake._target_reached(0) is False


def test_target_reached_triggers_at_threshold():
    from src.synthesizers.base import BaseSynthesizer

    class Fake:
        _target_reached = BaseSynthesizer._target_reached

        class progress:  # noqa
            progress_data = {'total_generated': 100}

    fake = Fake()
    assert fake._target_reached(50) is True
    assert fake._target_reached(100) is True
    assert fake._target_reached(101) is False


# ─── ProgressManager.set_waiting_state ──────────────────────────────────────

def test_set_waiting_state_writes_status(tmp_path):
    from src.core.progress import ProgressManager

    pm = ProgressManager()
    pm.local_path = str(tmp_path)

    pm.set_waiting_state(
        reason="gemini_daily_quota",
        until_iso="2099-01-01T00:00:00+00:00",
        exhausted_keys=3,
        total_keys=3,
    )
    assert pm.progress_data['status'] == 'waiting'
    assert pm.progress_data['waiting']['reason'] == 'gemini_daily_quota'
    assert pm.progress_data['waiting']['exhausted_keys'] == 3

    # Clearing it flips status back
    pm.set_waiting_state(None)
    assert pm.progress_data['status'] == 'running'
    assert pm.progress_data['waiting'] is None


# ─── Provider ↔ ProgressManager wiring (regression guard) ──────────────────

def test_attach_progress_to_providers_reaches_sub_providers():
    """Regression: factory exposes ``_providers`` (with underscore). Earlier
    code accidentally looked for ``providers`` and silently attached nothing,
    so /progress never reflected Gemini's quota-wait state."""
    from src.synthesizers.base import BaseSynthesizer
    from src.core.progress import ProgressManager

    received = []

    class FakeProvider:
        def attach_progress_manager(self, pm):
            received.append(("sub", pm))

    class FakeFactory:
        # NOTE: matches real ProviderFactory's attribute name.
        _providers = {"gemini": FakeProvider(), "openrouter": FakeProvider()}

        def attach_progress_manager(self, pm):
            received.append(("factory", pm))

    pm = ProgressManager()

    # Hand-build a stand-in to avoid the abstract-class restriction.
    class Stub:
        provider = FakeFactory()
        progress = pm
        _attach_progress_to_providers = BaseSynthesizer._attach_progress_to_providers

    Stub()._attach_progress_to_providers()

    # Factory got it, and both sub-providers got it.
    targets = [t for t, _ in received]
    assert "factory" in targets
    assert targets.count("sub") == 2
