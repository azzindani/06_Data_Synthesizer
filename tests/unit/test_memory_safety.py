"""Tests for memory-safety invariants — unbounded growth would silently
inflate RSS over the lifetime of a multi-day cron job."""

import json
from collections import deque

import pytest

pytestmark = pytest.mark.unit


# ─── ProgressManager.processed_items is a set with stable JSON output ──────

def test_processed_items_is_set_in_memory():
    from src.core.progress import ProgressManager
    pm = ProgressManager()
    assert isinstance(pm.progress_data['processed_items'], set)


def test_is_processed_o1_correctness():
    """O(1) set lookup must still answer correctly (regression for the
    list-to-set conversion)."""
    from src.core.progress import ProgressManager
    pm = ProgressManager()
    for i in range(1000):
        pm.mark_processed(f"item_{i}")
    assert pm.is_processed("item_0") is True
    assert pm.is_processed("item_999") is True
    assert pm.is_processed("item_1000") is False
    assert pm.progress_data['total_processed'] == 1000


def test_mark_processed_is_idempotent():
    """Re-marking shouldn't bump total_processed."""
    from src.core.progress import ProgressManager
    pm = ProgressManager()
    pm.mark_processed("dup_id")
    pm.mark_processed("dup_id")
    pm.mark_processed("dup_id")
    assert pm.progress_data['total_processed'] == 1


def test_progress_json_serialises_set_as_sorted_list(tmp_path):
    """JSON has no set type — we must emit a sorted list so the file
    is diffable across runs."""
    from src.core.progress import ProgressManager
    pm = ProgressManager()
    pm.local_path = str(tmp_path)
    pm.progress_file = "synthesis_progress.json"

    for item_id in ("item_c", "item_a", "item_b"):
        pm.mark_processed(item_id)

    pm.save(upload_to_hf=False)

    saved = json.loads((tmp_path / "synthesis_progress.json").read_text())
    assert saved['processed_items'] == ["item_a", "item_b", "item_c"]


def test_progress_json_load_round_trip(tmp_path):
    """Saving then loading must yield an equivalent set — covers the
    backward-compat list-in, set-out conversion."""
    from src.core.progress import ProgressManager

    pm1 = ProgressManager()
    pm1.local_path = str(tmp_path)
    pm1.progress_file = "synthesis_progress.json"
    for item_id in ("a", "b", "c"):
        pm1.mark_processed(item_id)
    pm1.save(upload_to_hf=False)

    pm2 = ProgressManager()
    pm2.local_path = str(tmp_path)
    pm2.progress_file = "synthesis_progress.json"
    # ProgressManager loads in __init__ when given config; here we call
    # explicitly because we constructed without a config.
    pm2.load()

    assert isinstance(pm2.progress_data['processed_items'], set)
    assert pm2.progress_data['processed_items'] == {"a", "b", "c"}
    assert pm2.is_processed("a")
    assert not pm2.is_processed("missing")


# ─── CostTracker.records is bounded ────────────────────────────────────────

def test_cost_tracker_records_capped():
    """Unbounded list → ~200 bytes per record × N requests. Cap with deque."""
    from src.utils.cost_tracker import CostTracker, _RECORDS_BUFFER_SIZE

    tracker = CostTracker()
    assert isinstance(tracker.records, deque)

    # Push more than the cap; size must not exceed it.
    overflow = _RECORDS_BUFFER_SIZE + 200
    for i in range(overflow):
        tracker.record_usage('gemini', 'gemini-flash', 100, 50)

    assert len(tracker.records) == _RECORDS_BUFFER_SIZE
    # Aggregates still count every call.
    assert tracker.totals['requests'] == overflow


def test_cost_tracker_get_recent_records_still_works():
    """Capping must not break the tail-slice API."""
    from src.utils.cost_tracker import CostTracker

    tracker = CostTracker()
    for i in range(50):
        tracker.record_usage('gemini', 'gemini-flash', i * 10, i * 5)

    recent = tracker.get_recent_records(limit=5)
    assert len(recent) == 5
    # The 5 most recent (last) records should be the highest input_tokens.
    assert recent[-1]['tokens'] > recent[0]['tokens']
