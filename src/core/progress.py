"""Progress tracking with persistence and HuggingFace sync."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .logger import get_logger


class ProgressManager:
    """Manages synthesis progress with local and remote persistence."""

    def __init__(self, config: Any = None):
        """Initialize progress manager.

        Args:
            config: Configuration object with output settings
        """
        self.logger = get_logger(__name__)
        self.config = config

        # Progress file paths
        if config:
            self.progress_file = config.output.progress_file
            self.repository = config.output.repository
            self.hf_token = config.output.huggingface_token
            self.local_path = config.output.local_path
        else:
            self.progress_file = "synthesis_progress.json"
            self.repository = ""
            self.hf_token = ""
            self.local_path = "./output"

        # Initialize progress data.
        # ``processed_items`` is held as a ``set`` in memory so the
        # is_processed() check used at the top of every synthesize_item is
        # O(1) instead of O(n). At 100k items the previous list-based "in"
        # check was the dominant CPU cost AND the cost was quadratic. The
        # set is serialised as a sorted list on save() for stable JSON.
        self.progress_data = {
            'processed_items': set(),
            'processed_topics': {},
            'current_item': None,
            'total_processed': 0,
            'total_generated': 0,
            'request_count': 0,
            'start_time': None,
            'last_update': None,
            'status': 'idle',          # idle | running | waiting | done
            'waiting': None,           # populated when sleeping on rate/quota
            'errors': [],
            'statistics': {
                'quality_counts': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0},
                'language_accuracy': {'passed': 0, 'failed': 0},
                'avg_score': 0.0,
                'provider_usage': {}
            }
        }

        # Load existing progress
        self.load()

    def load(self) -> None:
        """Load progress from storage (local first, then HF)."""
        loaded = False

        # Try local file first
        local_file = Path(self.local_path) / self.progress_file
        if local_file.exists():
            try:
                with open(local_file, 'r') as f:
                    saved = json.load(f)
                    self._merge_loaded(saved)
                self.logger.info(f"Progress loaded from local: {self.progress_data['total_processed']} items")
                loaded = True
            except Exception as e:
                self.logger.warning(f"Failed to load local progress: {e}")

        # Try HuggingFace if not loaded locally
        if not loaded and self.repository and self.hf_token:
            try:
                from huggingface_hub import hf_hub_download
                progress_path = hf_hub_download(
                    repo_id=self.repository,
                    filename=self.progress_file,
                    repo_type="dataset",
                    token=self.hf_token
                )
                with open(progress_path, 'r') as f:
                    saved = json.load(f)
                    self._merge_loaded(saved)
                self.logger.info(f"Progress loaded from HF: {self.progress_data['total_processed']} items")
                loaded = True
            except Exception as e:
                self.logger.debug(f"No HF progress found: {e}")

        # Initialize fresh if nothing loaded
        if not loaded:
            self.progress_data['start_time'] = datetime.now().isoformat()
            self.logger.info("Starting fresh progress tracking")

    def _merge_loaded(self, saved: Dict) -> None:
        """Merge a freshly-loaded JSON blob into ``progress_data``.

        Converts the JSON ``processed_items`` list back to a set (the
        on-disk format must be a list since JSON has no set type).
        """
        if 'processed_items' in saved and isinstance(saved['processed_items'], list):
            saved = dict(saved)  # don't mutate caller's dict
            saved['processed_items'] = set(saved['processed_items'])
        self.progress_data.update(saved)

    def save(self, upload_to_hf: bool = True) -> None:
        """Save progress to storage.

        Args:
            upload_to_hf: Whether to upload to HuggingFace
        """
        self.progress_data['last_update'] = datetime.now().isoformat()

        # Serialise the processed_items set as a sorted list so the JSON
        # output is stable (diffable across runs). Sort once per save —
        # negligible vs the file write itself.
        snapshot = dict(self.progress_data)
        items = snapshot.get('processed_items')
        if isinstance(items, set):
            snapshot['processed_items'] = sorted(items)

        # Convert numpy/pandas types to native Python
        clean_data = self._convert_types(snapshot)

        # Save locally
        local_dir = Path(self.local_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / self.progress_file

        try:
            with open(local_file, 'w') as f:
                json.dump(clean_data, f, indent=2)
            self.logger.debug("Progress saved locally")
        except Exception as e:
            self.logger.error(f"Failed to save local progress: {e}")

        # Upload to HuggingFace
        if upload_to_hf and self.repository and self.hf_token:
            try:
                from huggingface_hub import upload_file
                upload_file(
                    path_or_fileobj=str(local_file),
                    path_in_repo=self.progress_file,
                    repo_id=self.repository,
                    repo_type="dataset",
                    token=self.hf_token,
                    commit_message=f"Progress: {self.progress_data['total_processed']} items"
                )
                self.logger.debug("Progress saved to HuggingFace")
            except Exception as e:
                self.logger.warning(f"Failed to upload progress to HF: {e}")

    def _convert_types(self, obj: Any) -> Any:
        """Convert numpy/pandas types to native Python types."""
        if isinstance(obj, dict):
            return {k: self._convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_types(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        return obj

    def is_processed(self, item_id: str) -> bool:
        """Check if an item has been processed.

        Args:
            item_id: Unique identifier for the item

        Returns:
            True if already processed
        """
        return item_id in self.progress_data['processed_items']

    def mark_processed(self, item_id: str, stats: Optional[Dict] = None) -> None:
        """Mark an item as processed.

        Args:
            item_id: Unique identifier for the item
            stats: Optional statistics about the processed item
        """
        items = self.progress_data['processed_items']
        # Set.add is O(1) and idempotent — no need for an "in" check first.
        before = len(items)
        items.add(item_id)
        if len(items) > before:
            self.progress_data['total_processed'] += 1

        if stats:
            self._update_statistics(stats)

    def mark_topic_progress(self, topic: str, count: int, total: int) -> None:
        """Update progress for a specific topic.

        Args:
            topic: Topic name
            count: Number completed
            total: Total expected
        """
        self.progress_data['processed_topics'][topic] = {
            'completed': count,
            'total': total,
            'percentage': (count / total * 100) if total > 0 else 0
        }

    def add_generated(self, count: int) -> None:
        """Add to total generated count.

        Args:
            count: Number of items generated
        """
        self.progress_data['total_generated'] += count

    def increment_requests(self) -> int:
        """Increment and return request count."""
        self.progress_data['request_count'] += 1
        return self.progress_data['request_count']

    def add_error(self, error_data: Dict) -> None:
        """Log an error.

        Args:
            error_data: Error information dictionary
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            **error_data
        }
        self.progress_data['errors'].append(error_entry)

        # Keep only last 100 errors
        if len(self.progress_data['errors']) > 100:
            self.progress_data['errors'] = self.progress_data['errors'][-100:]

    def _update_statistics(self, stats: Dict) -> None:
        """Update running statistics."""
        current = self.progress_data['statistics']

        # Update quality counts
        for quality, count in stats.get('quality_counts', {}).items():
            if quality in current['quality_counts']:
                current['quality_counts'][quality] += count

        # Update language accuracy
        if 'language_passed' in stats:
            current['language_accuracy']['passed'] += stats['language_passed']
        if 'language_failed' in stats:
            current['language_accuracy']['failed'] += stats['language_failed']

        # Update provider usage
        provider = stats.get('provider')
        if provider:
            if provider not in current['provider_usage']:
                current['provider_usage'][provider] = 0
            current['provider_usage'][provider] += stats.get('requests', 1)

    def get_statistics(self) -> Dict:
        """Get current statistics."""
        return self.progress_data['statistics']

    def set_waiting_state(self, reason: Optional[str] = None,
                          until_iso: Optional[str] = None,
                          **extra) -> None:
        """Mark the run as waiting on an external resource (e.g. quota reset).

        Pass ``reason=None`` to clear (synthesis resumes). The ``until_iso``
        timestamp is what /progress exposes so external monitors know when
        to check back.

        Args:
            reason: short tag, e.g. ``"gemini_daily_quota"``. ``None`` clears.
            until_iso: ISO 8601 timestamp of expected resume.
            **extra: any extra fields to attach to the waiting record.
        """
        if reason is None:
            self.progress_data['status'] = 'running'
            self.progress_data['waiting'] = None
        else:
            self.progress_data['status'] = 'waiting'
            self.progress_data['waiting'] = {
                'reason': reason,
                'until': until_iso,
                'since': datetime.now().isoformat(),
                **extra,
            }
        try:
            # Persist immediately so /progress reflects state during a long wait.
            self.save(upload_to_hf=False)
        except Exception as e:
            self.logger.debug(f"Could not persist waiting state: {e}")

    def show_progress(self, total_expected: int = 0) -> None:
        """Display current progress.

        Args:
            total_expected: Total items expected
        """
        processed = self.progress_data['total_processed']
        generated = self.progress_data['total_generated']
        requests = self.progress_data['request_count']

        if total_expected > 0:
            pct = (processed / total_expected) * 100
            print(f"Progress: {processed:,}/{total_expected:,} ({pct:.1f}%)")
        else:
            print(f"Processed: {processed:,} items")

        print(f"Generated: {generated:,} outputs")
        print(f"API Requests: {requests:,}")

        # Show topic progress
        topics = self.progress_data.get('processed_topics', {})
        if topics:
            completed = sum(1 for t in topics.values() if t.get('completed', 0) >= t.get('total', 1))
            print(f"Topics: {completed}/{len(topics)} completed")

        # Show quality distribution
        quality = self.progress_data['statistics']['quality_counts']
        total_quality = sum(quality.values())
        if total_quality > 0:
            print(f"Quality: {quality}")

    def get_resume_point(self) -> Optional[str]:
        """Get a processed item for resuming.

        ``processed_items`` is a set (unordered) so we return the
        lexicographically last element rather than insertion-order last.
        The synthesizer doesn't depend on insertion order — it uses
        ``is_processed()`` to skip already-done items regardless.
        """
        items = self.progress_data['processed_items']
        if not items:
            return None
        return max(items) if isinstance(items, set) else items[-1]


if __name__ == "__main__":
    print("=" * 60)
    print("PROGRESS MANAGER TEST")
    print("=" * 60)

    # Test 1: Create progress manager without config
    pm = ProgressManager()
    print("  ✓ Progress manager created")

    # Test 2: Mark items as processed
    pm.mark_processed("item_001", {'quality_counts': {'good': 1}})
    pm.mark_processed("item_002", {'quality_counts': {'excellent': 1}})
    pm.mark_processed("item_003")
    print(f"  ✓ Marked 3 items processed: {pm.progress_data['total_processed']}")

    # Test 3: Check if processed
    assert pm.is_processed("item_001") == True
    assert pm.is_processed("item_999") == False
    print("  ✓ is_processed() works correctly")

    # Test 4: Topic progress
    pm.mark_topic_progress("Topic A", 10, 50)
    pm.mark_topic_progress("Topic B", 50, 50)
    print("  ✓ Topic progress updated")

    # Test 5: Error logging
    pm.add_error({'type': 'api_error', 'message': 'Rate limited'})
    print(f"  ✓ Error logged: {len(pm.progress_data['errors'])} errors")

    # Test 6: Increment requests
    for _ in range(5):
        pm.increment_requests()
    print(f"  ✓ Request count: {pm.progress_data['request_count']}")

    # Test 7: Show progress
    print("\nProgress Summary:")
    pm.show_progress(100)

    # Test 8: Save locally (no HF)
    pm.save(upload_to_hf=False)
    print("  ✓ Progress saved locally")

    # Test 9: Get resume point
    resume = pm.get_resume_point()
    print(f"  ✓ Resume point: {resume}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
