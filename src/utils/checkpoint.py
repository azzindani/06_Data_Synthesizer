"""Checkpoint management for resuming from specific points."""

import os
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..core.logger import get_logger


class CheckpointManager:
    """Manages checkpoints for resuming synthesis from specific points."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.logger = get_logger(__name__)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, name: str, state: Dict[str, Any],
                        metadata: Dict[str, Any] = None) -> str:
        """Save a checkpoint.

        Args:
            name: Checkpoint name/identifier
            state: State to save
            metadata: Optional metadata

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = self.checkpoint_dir / filename

        checkpoint_data = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'metadata': metadata or {}
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved checkpoint: {filepath}")
        return str(filepath)

    def load_checkpoint(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint by name or path.

        Args:
            identifier: Checkpoint name, path, or 'latest'

        Returns:
            Checkpoint data or None
        """
        # Check if it's a direct path
        if os.path.exists(identifier):
            filepath = identifier
        elif identifier == 'latest':
            filepath = self._get_latest_checkpoint()
        else:
            filepath = self._find_checkpoint_by_name(identifier)

        if not filepath:
            self.logger.warning(f"Checkpoint not found: {identifier}")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.info(f"Loaded checkpoint: {filepath}")
            return data

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get the most recent checkpoint.

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if checkpoints:
            return str(checkpoints[0])
        return None

    def _find_checkpoint_by_name(self, name: str) -> Optional[str]:
        """Find checkpoint by name prefix.

        Args:
            name: Checkpoint name prefix

        Returns:
            Path to checkpoint or None
        """
        # Find all matching checkpoints
        pattern = f"{name}*.json"
        matches = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if matches:
            return str(matches[0])
        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.

        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []

        for filepath in sorted(self.checkpoint_dir.glob("*.json"), reverse=True):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                checkpoints.append({
                    'path': str(filepath),
                    'name': data.get('name', filepath.stem),
                    'timestamp': data.get('timestamp', ''),
                    'metadata': data.get('metadata', {})
                })
            except Exception:
                pass

        return checkpoints

    def delete_checkpoint(self, identifier: str) -> bool:
        """Delete a checkpoint.

        Args:
            identifier: Checkpoint name or path

        Returns:
            True if deleted
        """
        if os.path.exists(identifier):
            filepath = identifier
        else:
            filepath = self._find_checkpoint_by_name(identifier)

        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            self.logger.info(f"Deleted checkpoint: {filepath}")
            return True

        return False

    def cleanup_old_checkpoints(self, keep: int = 10):
        """Remove old checkpoints, keeping only the most recent.

        Args:
            keep: Number of checkpoints to keep
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for checkpoint in checkpoints[keep:]:
            checkpoint.unlink()
            self.logger.debug(f"Removed old checkpoint: {checkpoint}")

        if len(checkpoints) > keep:
            self.logger.info(f"Cleaned up {len(checkpoints) - keep} old checkpoints")


class ResumableTask:
    """Wrapper for tasks that can be resumed from checkpoint."""

    def __init__(self, task_name: str, checkpoint_manager: CheckpointManager = None):
        """Initialize resumable task.

        Args:
            task_name: Name of the task
            checkpoint_manager: Checkpoint manager instance
        """
        self.logger = get_logger(__name__)
        self.task_name = task_name
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()

        # Task state
        self._state = {
            'processed_items': [],
            'current_index': 0,
            'statistics': {},
            'errors': []
        }

    def resume_from_checkpoint(self, identifier: str = 'latest') -> bool:
        """Resume task from a checkpoint.

        Args:
            identifier: Checkpoint identifier

        Returns:
            True if resumed successfully
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(identifier)

        if not checkpoint:
            return False

        self._state = checkpoint.get('state', self._state)
        self.logger.info(f"Resumed from checkpoint: {len(self._state['processed_items'])} items processed")
        return True

    def save_checkpoint(self, metadata: Dict[str, Any] = None) -> str:
        """Save current task state as checkpoint.

        Args:
            metadata: Additional metadata

        Returns:
            Checkpoint path
        """
        return self.checkpoint_manager.save_checkpoint(
            self.task_name,
            self._state,
            metadata
        )

    def mark_processed(self, item_id: str):
        """Mark an item as processed.

        Args:
            item_id: Item identifier
        """
        if item_id not in self._state['processed_items']:
            self._state['processed_items'].append(item_id)
            self._state['current_index'] += 1

    def is_processed(self, item_id: str) -> bool:
        """Check if item was already processed.

        Args:
            item_id: Item identifier

        Returns:
            True if already processed
        """
        return item_id in self._state['processed_items']

    def get_remaining_items(self, all_items: List[str]) -> List[str]:
        """Get items that haven't been processed yet.

        Args:
            all_items: List of all item IDs

        Returns:
            List of remaining item IDs
        """
        processed = set(self._state['processed_items'])
        return [item for item in all_items if item not in processed]

    def update_statistics(self, key: str, value: Any):
        """Update task statistics.

        Args:
            key: Statistic key
            value: Statistic value
        """
        self._state['statistics'][key] = value

    def record_error(self, error: str, item_id: str = None):
        """Record an error.

        Args:
            error: Error message
            item_id: Related item ID
        """
        self._state['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'item_id': item_id
        })

    @property
    def processed_count(self) -> int:
        """Get number of processed items."""
        return len(self._state['processed_items'])

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get task statistics."""
        return self._state['statistics'].copy()

    @property
    def errors(self) -> List[Dict[str, Any]]:
        """Get recorded errors."""
        return self._state['errors'].copy()


if __name__ == "__main__":
    print("=" * 60)
    print("CHECKPOINT MANAGER TEST")
    print("=" * 60)

    import tempfile
    import shutil

    # Create temporary directory
    tmpdir = tempfile.mkdtemp()

    try:
        # Test 1: Create manager
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        print(f"  ✓ Created checkpoint manager")

        # Test 2: Save checkpoint
        state = {
            'processed_items': ['item_001', 'item_002'],
            'current_index': 2,
            'statistics': {'total': 100}
        }
        path = manager.save_checkpoint('test_task', state, {'version': '1.0'})
        print(f"  ✓ Saved checkpoint: {path}")

        # Test 3: List checkpoints
        checkpoints = manager.list_checkpoints()
        print(f"  ✓ Listed checkpoints: {len(checkpoints)}")

        # Test 4: Load checkpoint
        loaded = manager.load_checkpoint('latest')
        print(f"  ✓ Loaded checkpoint")
        assert loaded['state'] == state

        # Test 5: ResumableTask
        task = ResumableTask('synthesis', manager)
        task.mark_processed('item_001')
        task.mark_processed('item_002')
        task.update_statistics('generated', 50)
        print(f"  ✓ Created resumable task")

        # Test 6: Save and resume
        task.save_checkpoint()

        task2 = ResumableTask('synthesis', manager)
        task2.resume_from_checkpoint('latest')
        print(f"  ✓ Resumed task: {task2.processed_count} items")

        # Test 7: Get remaining items
        all_items = ['item_001', 'item_002', 'item_003', 'item_004']
        remaining = task2.get_remaining_items(all_items)
        print(f"  ✓ Remaining items: {remaining}")

    finally:
        shutil.rmtree(tmpdir)

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
