"""Unit tests for Phase 4 components."""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch


class TestInstanceCoordinator:
    """Tests for multi-instance coordinator."""

    def test_initialization(self):
        """Test coordinator initialization."""
        from src.utils.coordinator import InstanceCoordinator

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = InstanceCoordinator(lock_dir=tmpdir)
            assert coord.instance_id is not None
            assert (coord.lock_dir / "instance_registry.json").exists()

    def test_acquire_item(self):
        """Test acquiring lock on item."""
        from src.utils.coordinator import InstanceCoordinator

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = InstanceCoordinator(lock_dir=tmpdir)

            # Should acquire successfully
            result = coord.acquire_item("test_001")
            assert result is True

            # Same instance can re-acquire
            result2 = coord.acquire_item("test_001")
            assert result2 is True

    def test_release_item(self):
        """Test releasing lock on item."""
        from src.utils.coordinator import InstanceCoordinator

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = InstanceCoordinator(lock_dir=tmpdir)

            coord.acquire_item("test_001")
            coord.release_item("test_001")

            # Lock file should be removed
            lock_file = coord.lock_dir / "item_test_001.lock"
            assert not lock_file.exists()

    def test_mark_completed(self):
        """Test marking item as completed."""
        from src.utils.coordinator import InstanceCoordinator

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = InstanceCoordinator(lock_dir=tmpdir)

            coord.acquire_item("test_001")
            coord.mark_completed("test_001")

            assert coord.is_completed("test_001") is True
            assert coord.is_completed("test_002") is False

    def test_heartbeat(self):
        """Test heartbeat update."""
        from src.utils.coordinator import InstanceCoordinator

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = InstanceCoordinator(lock_dir=tmpdir)
            coord.heartbeat({'processed': 10})

            # Check registry was updated
            with open(coord._registry_file) as f:
                registry = json.load(f)

            assert coord.instance_id in registry['instances']
            assert registry['instances'][coord.instance_id]['processed'] == 10

    def test_get_active_instances(self):
        """Test getting active instances."""
        from src.utils.coordinator import InstanceCoordinator

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = InstanceCoordinator(lock_dir=tmpdir)

            active = coord.get_active_instances()
            assert len(active) == 1
            assert active[0]['instance_id'] == coord.instance_id

    def test_deregister(self):
        """Test deregistering instance."""
        from src.utils.coordinator import InstanceCoordinator

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = InstanceCoordinator(lock_dir=tmpdir)
            coord.deregister()

            with open(coord._registry_file) as f:
                registry = json.load(f)

            assert coord.instance_id not in registry['instances']


class TestWorkDistributor:
    """Tests for work distributor."""

    def test_get_next_item(self):
        """Test getting next available item."""
        from src.utils.coordinator import InstanceCoordinator, WorkDistributor

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = InstanceCoordinator(lock_dir=tmpdir)
            dist = WorkDistributor(coord)

            items = ['item_001', 'item_002', 'item_003']
            next_item = dist.get_next_item(items)

            assert next_item == 'item_001'

    def test_skip_completed_items(self):
        """Test that completed items are skipped."""
        from src.utils.coordinator import InstanceCoordinator, WorkDistributor

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = InstanceCoordinator(lock_dir=tmpdir)
            dist = WorkDistributor(coord)

            # Mark first item as completed
            coord.mark_completed('item_001')

            items = ['item_001', 'item_002', 'item_003']
            next_item = dist.get_next_item(items)

            assert next_item == 'item_002'

    def test_distribute_items(self):
        """Test distributing batch of items."""
        from src.utils.coordinator import InstanceCoordinator, WorkDistributor

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = InstanceCoordinator(lock_dir=tmpdir)
            dist = WorkDistributor(coord)

            items = [f'item_{i:03d}' for i in range(10)]
            acquired = dist.distribute_items(items, batch_size=3)

            assert len(acquired) == 3


class TestFileLock:
    """Tests for file locking."""

    def test_basic_lock(self):
        """Test basic file locking."""
        from src.utils.coordinator import FileLock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = os.path.join(tmpdir, "test.lock")

            with FileLock(lock_file):
                # Lock acquired, file should exist
                assert os.path.exists(lock_file)

    def test_lock_timeout(self):
        """Test lock timeout."""
        from src.utils.coordinator import FileLock
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = os.path.join(tmpdir, "test.lock")

            # Hold lock in another thread
            def hold_lock():
                with FileLock(lock_file, timeout=10):
                    import time
                    time.sleep(2)

            thread = threading.Thread(target=hold_lock)
            thread.start()

            import time
            time.sleep(0.1)  # Let thread acquire lock

            # Try to acquire with short timeout
            with pytest.raises(BlockingIOError):
                with FileLock(lock_file, timeout=0.1):
                    pass

            thread.join()


class TestDashboard:
    """Tests for dashboard server."""

    def test_dashboard_creation(self):
        """Test dashboard creation."""
        from src.dashboard import create_dashboard

        dashboard = create_dashboard(port=8081)
        assert dashboard.port == 8081

    def test_add_event(self):
        """Test adding events."""
        from src.dashboard import DashboardServer

        dashboard = DashboardServer(port=8082)
        dashboard.add_event("Test event")

        events = dashboard._get_recent_events()
        assert len(events) == 1
        assert events[0]['message'] == "Test event"

    def test_status(self):
        """Test status retrieval."""
        from src.dashboard import DashboardServer

        dashboard = DashboardServer(port=8083)
        status = dashboard._get_status()

        assert 'running' in status
        assert 'uptime' in status


class TestBenchmarks:
    """Tests for benchmark functions."""

    def test_json_parsing_benchmark(self):
        """Test JSON parsing benchmark."""
        # Import the benchmark function
        import sys
        sys.path.insert(0, 'scripts')
        from benchmark import benchmark_json_parsing

        results = benchmark_json_parsing(num_iterations=10)

        assert 'iterations' in results
        assert 'mean_us' in results
        assert results['iterations'] == 40  # 10 iterations * 4 test cases

    def test_memory_benchmark(self):
        """Test memory benchmark."""
        import sys
        sys.path.insert(0, 'scripts')
        from benchmark import benchmark_memory_usage

        results = benchmark_memory_usage()

        assert 'current_mb' in results
        assert 'peak_mb' in results
        assert results['peak_mb'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
