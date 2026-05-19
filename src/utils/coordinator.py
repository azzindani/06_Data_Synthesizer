"""Multi-instance coordination using file-based locking.

The locking primitive (``FileLock`` below) uses ``fcntl.flock`` on POSIX
systems and ``msvcrt.locking`` on Windows so the same API works in CI on
all three OSes — the synthesizer is primarily run inside a Linux container,
but the unit tests must import cleanly on Windows runners too.
"""

import os
import sys
import json
import time
import socket
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from ..core.logger import get_logger

_IS_WINDOWS = sys.platform == "win32"
if _IS_WINDOWS:
    import msvcrt  # type: ignore[import]

    def _acquire(fd):
        # msvcrt locks N bytes from the current file position; rewind so we
        # lock byte 0 (consistent target regardless of file contents).
        fd.seek(0)
        msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)

    def _release(fd):
        fd.seek(0)
        msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl  # type: ignore[import]

    def _acquire(fd):
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release(fd):
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)


class InstanceCoordinator:
    """Coordinates multiple synthesis instances to avoid duplicate work."""

    def __init__(self, lock_dir: str = ".locks", instance_id: str = None):
        """Initialize coordinator.

        Args:
            lock_dir: Directory for lock files
            instance_id: Unique identifier for this instance
        """
        self.logger = get_logger(__name__)
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        # Generate instance ID
        self.instance_id = instance_id or f"{socket.gethostname()}_{os.getpid()}"
        self.logger.info(f"Instance ID: {self.instance_id}")

        # State files
        self._registry_file = self.lock_dir / "instance_registry.json"
        self._work_queue_file = self.lock_dir / "work_queue.json"

        # Register this instance
        self._register_instance()

    def _register_instance(self):
        """Register this instance in the registry."""
        with self._file_lock(self._registry_file):
            registry = self._load_json(self._registry_file, default={'instances': {}})

            registry['instances'][self.instance_id] = {
                'started': datetime.now().isoformat(),
                'last_heartbeat': datetime.now().isoformat(),
                'status': 'active',
                'processed': 0
            }

            self._save_json(self._registry_file, registry)
            self.logger.info(f"Registered instance: {self.instance_id}")

    def heartbeat(self, stats: Dict[str, Any] = None):
        """Update heartbeat for this instance.

        Args:
            stats: Optional statistics to include
        """
        with self._file_lock(self._registry_file):
            registry = self._load_json(self._registry_file, default={'instances': {}})

            if self.instance_id in registry['instances']:
                registry['instances'][self.instance_id]['last_heartbeat'] = datetime.now().isoformat()
                if stats:
                    registry['instances'][self.instance_id].update(stats)

            self._save_json(self._registry_file, registry)

    def acquire_item(self, item_id: str, timeout: int = 300) -> bool:
        """Acquire lock on an item for processing.

        Args:
            item_id: Item identifier
            timeout: Lock timeout in seconds

        Returns:
            True if acquired, False if already locked
        """
        lock_file = self.lock_dir / f"item_{item_id}.lock"

        try:
            with self._file_lock(lock_file, timeout=0.1):
                # Check if already locked by another instance
                lock_info = self._load_json(lock_file, default={})

                if lock_info:
                    # Check if lock expired
                    locked_at = datetime.fromisoformat(lock_info.get('locked_at', '2000-01-01'))
                    if datetime.now() - locked_at < timedelta(seconds=timeout):
                        if lock_info.get('instance_id') != self.instance_id:
                            return False

                # Acquire lock
                self._save_json(lock_file, {
                    'instance_id': self.instance_id,
                    'locked_at': datetime.now().isoformat(),
                    'item_id': item_id
                })

                self.logger.debug(f"Acquired lock: {item_id}")
                return True

        except BlockingIOError:
            return False

    def release_item(self, item_id: str):
        """Release lock on an item.

        Args:
            item_id: Item identifier
        """
        lock_file = self.lock_dir / f"item_{item_id}.lock"

        if lock_file.exists():
            lock_file.unlink()
            self.logger.debug(f"Released lock: {item_id}")

    def mark_completed(self, item_id: str):
        """Mark item as completed.

        Args:
            item_id: Item identifier
        """
        completed_file = self.lock_dir / "completed.json"

        with self._file_lock(completed_file):
            completed = self._load_json(completed_file, default={'items': []})
            if item_id not in completed['items']:
                completed['items'].append(item_id)
            self._save_json(completed_file, completed)

        self.release_item(item_id)

    def is_completed(self, item_id: str) -> bool:
        """Check if item is already completed.

        Args:
            item_id: Item identifier

        Returns:
            True if completed
        """
        completed_file = self.lock_dir / "completed.json"
        completed = self._load_json(completed_file, default={'items': []})
        return item_id in completed['items']

    def get_active_instances(self) -> List[Dict[str, Any]]:
        """Get list of active instances.

        Returns:
            List of instance information
        """
        registry = self._load_json(self._registry_file, default={'instances': {}})
        active = []

        for instance_id, info in registry['instances'].items():
            last_heartbeat = datetime.fromisoformat(info.get('last_heartbeat', '2000-01-01'))
            if datetime.now() - last_heartbeat < timedelta(minutes=5):
                active.append({
                    'instance_id': instance_id,
                    **info
                })

        return active

    def cleanup_stale_locks(self, timeout: int = 600):
        """Clean up stale lock files.

        Args:
            timeout: Consider locks stale after this many seconds
        """
        for lock_file in self.lock_dir.glob("item_*.lock"):
            try:
                lock_info = self._load_json(lock_file, default={})
                if lock_info:
                    locked_at = datetime.fromisoformat(lock_info.get('locked_at', '2000-01-01'))
                    if datetime.now() - locked_at > timedelta(seconds=timeout):
                        lock_file.unlink()
                        self.logger.info(f"Cleaned up stale lock: {lock_file.name}")
            except Exception as e:
                self.logger.warning(f"Error cleaning lock {lock_file}: {e}")

    def deregister(self):
        """Deregister this instance."""
        with self._file_lock(self._registry_file):
            registry = self._load_json(self._registry_file, default={'instances': {}})
            if self.instance_id in registry['instances']:
                del registry['instances'][self.instance_id]
            self._save_json(self._registry_file, registry)

        self.logger.info(f"Deregistered instance: {self.instance_id}")

    def _file_lock(self, filepath: Path, timeout: float = 10):
        """Context manager for file locking."""
        return FileLock(filepath, timeout)

    def _load_json(self, filepath: Path, default: Any = None) -> Any:
        """Load JSON from file."""
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return default or {}

    def _save_json(self, filepath: Path, data: Any):
        """Save JSON to file."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class FileLock:
    """Context manager for file-based locking."""

    def __init__(self, filepath: Path, timeout: float = 10):
        """Initialize file lock.

        Args:
            filepath: Path to lock file
            timeout: Lock acquisition timeout
        """
        self.filepath = Path(filepath)
        self.timeout = timeout
        self._fd = None

    def __enter__(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.filepath, 'a+')

        start_time = time.time()
        while True:
            try:
                _acquire(self._fd)
                return self
            except (BlockingIOError, OSError) as e:
                # Windows raises OSError(EACCES/EAGAIN); POSIX raises
                # BlockingIOError. Treat both as "still locked".
                if not _IS_WINDOWS and not isinstance(e, BlockingIOError):
                    raise
                if time.time() - start_time >= self.timeout:
                    raise
                time.sleep(0.01)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fd:
            try:
                _release(self._fd)
            except OSError:
                # Best-effort unlock — file may have been closed already.
                pass
            self._fd.close()
        return False


class WorkDistributor:
    """Distributes work items among instances."""

    def __init__(self, coordinator: InstanceCoordinator):
        """Initialize work distributor.

        Args:
            coordinator: Instance coordinator
        """
        self.coordinator = coordinator
        self.logger = get_logger(__name__)

    def get_next_item(self, items: List[str]) -> Optional[str]:
        """Get next available item for processing.

        Args:
            items: List of all item IDs

        Returns:
            Item ID or None if no items available
        """
        for item_id in items:
            # Skip completed items
            if self.coordinator.is_completed(item_id):
                continue

            # Try to acquire lock
            if self.coordinator.acquire_item(item_id):
                return item_id

        return None

    def distribute_items(self, items: List[str], batch_size: int = 10) -> List[str]:
        """Get batch of items for this instance.

        Args:
            items: List of all item IDs
            batch_size: Number of items to acquire

        Returns:
            List of acquired item IDs
        """
        acquired = []

        for item_id in items:
            if len(acquired) >= batch_size:
                break

            if self.coordinator.is_completed(item_id):
                continue

            if self.coordinator.acquire_item(item_id):
                acquired.append(item_id)

        self.logger.info(f"Acquired {len(acquired)} items for processing")
        return acquired


if __name__ == "__main__":
    print("=" * 60)
    print("COORDINATOR TEST")
    print("=" * 60)

    import tempfile
    import shutil

    # Create temporary lock directory
    tmpdir = tempfile.mkdtemp()

    try:
        # Test 1: Create coordinator
        coord = InstanceCoordinator(lock_dir=tmpdir)
        print(f"  ✓ Created coordinator: {coord.instance_id}")

        # Test 2: Acquire item
        acquired = coord.acquire_item("test_001")
        print(f"  ✓ Acquired item: {acquired}")

        # Test 3: Check if item is locked
        acquired2 = coord.acquire_item("test_001")
        print(f"  ✓ Re-acquire blocked: {not acquired2}")

        # Test 4: Mark completed
        coord.mark_completed("test_001")
        is_done = coord.is_completed("test_001")
        print(f"  ✓ Item completed: {is_done}")

        # Test 5: Heartbeat
        coord.heartbeat({'processed': 1})
        print(f"  ✓ Heartbeat sent")

        # Test 6: Get active instances
        active = coord.get_active_instances()
        print(f"  ✓ Active instances: {len(active)}")

        # Test 7: Deregister
        coord.deregister()
        print(f"  ✓ Deregistered")

    finally:
        shutil.rmtree(tmpdir)

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
