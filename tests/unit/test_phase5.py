"""Unit tests for Phase 5 components."""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, MagicMock


class TestLoadBalancer:
    """Tests for dynamic load balancer."""

    def test_initialization(self):
        """Test load balancer initialization."""
        from src.providers.load_balancer import LoadBalancer
        from src.providers.base import BaseProvider

        providers = {
            'gemini': Mock(spec=BaseProvider),
            'openai': Mock(spec=BaseProvider)
        }

        balancer = LoadBalancer(providers)
        assert len(balancer._providers) == 2

    def test_get_provider_adaptive(self):
        """Test adaptive provider selection."""
        from src.providers.load_balancer import LoadBalancer

        providers = {
            'gemini': Mock(),
            'openai': Mock()
        }

        balancer = LoadBalancer(providers, strategy="adaptive")
        name, provider = balancer.get_provider()
        assert name in providers

    def test_get_provider_round_robin(self):
        """Test round robin selection."""
        from src.providers.load_balancer import LoadBalancer

        providers = {
            'gemini': Mock(),
            'openai': Mock()
        }

        balancer = LoadBalancer(providers, strategy="round_robin")

        # Get providers in sequence
        names = []
        for _ in range(4):
            name, _ = balancer.get_provider()
            names.append(name)

        # Should alternate
        assert len(set(names)) == 2

    def test_record_result(self):
        """Test recording results."""
        from src.providers.load_balancer import LoadBalancer

        providers = {'gemini': Mock()}
        balancer = LoadBalancer(providers)

        balancer.record_result('gemini', True, 500)
        balancer.record_result('gemini', True, 600)
        balancer.record_result('gemini', False, 0)

        metrics = balancer.get_metrics()
        assert metrics['gemini']['total_requests'] == 3
        assert metrics['gemini']['successful_requests'] == 2
        assert metrics['gemini']['failed_requests'] == 1

    def test_metrics_calculation(self):
        """Test metrics calculation."""
        from src.providers.load_balancer import ProviderMetrics

        metrics = ProviderMetrics(name='test')
        metrics.total_requests = 10
        metrics.successful_requests = 8
        metrics.failed_requests = 2
        metrics.total_latency_ms = 4000

        assert metrics.success_rate == 0.8
        assert metrics.avg_latency_ms == 500
        assert metrics.score > 0

    def test_set_strategy(self):
        """Test changing strategy."""
        from src.providers.load_balancer import LoadBalancer

        providers = {'gemini': Mock()}
        balancer = LoadBalancer(providers)

        balancer.set_strategy('weighted')
        assert balancer._strategy == 'weighted'

        with pytest.raises(ValueError):
            balancer.set_strategy('invalid')


class TestPromptTemplate:
    """Tests for prompt templates."""

    def test_template_creation(self):
        """Test template creation."""
        from src.utils.templates import PromptTemplate

        template = PromptTemplate(
            template="Hello ${name}!",
            name="greeting",
            description="A greeting"
        )

        assert template.name == "greeting"
        assert 'name' in template.variables

    def test_template_render(self):
        """Test template rendering."""
        from src.utils.templates import PromptTemplate

        template = PromptTemplate("Generate ${count} items about ${topic}")
        result = template.render(count=5, topic="science")

        assert "5" in result
        assert "science" in result

    def test_template_validate(self):
        """Test template validation."""
        from src.utils.templates import PromptTemplate

        template = PromptTemplate("${a} and ${b}")

        missing = template.validate(a="value")
        assert 'b' in missing
        assert 'a' not in missing


class TestTemplateManager:
    """Tests for template manager."""

    def test_initialization(self):
        """Test manager initialization."""
        from src.utils.templates import TemplateManager

        manager = TemplateManager()
        assert len(manager.list_templates()) > 0

    def test_get_builtin_template(self):
        """Test getting built-in template."""
        from src.utils.templates import TemplateManager

        manager = TemplateManager()
        template = manager.get_template('qa_synthesis')

        assert template is not None
        assert 'num_variants' in template.variables

    def test_render_template(self):
        """Test rendering template."""
        from src.utils.templates import TemplateManager

        manager = TemplateManager()
        result = manager.render(
            'qa_synthesis',
            num_variants=3,
            topic="Test",
            language="English",
            min_length=50,
            max_length=100
        )

        assert "3" in result
        assert "Test" in result

    def test_create_prompt(self):
        """Test creating full prompt."""
        from src.utils.templates import TemplateManager

        manager = TemplateManager()
        system, user = manager.create_prompt(
            'qa_synthesis',
            'qa_expert',
            num_variants=5,
            topic="Test",
            language="English",
            min_length=50,
            max_length=100
        )

        assert len(system) > 0
        assert len(user) > 0

    def test_save_template(self):
        """Test saving custom template."""
        from src.utils.templates import TemplateManager

        manager = TemplateManager()
        manager.save_template('custom', 'Test ${var}', 'Custom template')

        template = manager.get_template('custom')
        assert template is not None


class TestCheckpointManager:
    """Tests for checkpoint manager."""

    def test_initialization(self):
        """Test manager initialization."""
        from src.utils.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            assert manager.checkpoint_dir.exists()

    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        from src.utils.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            state = {'processed': 10, 'items': ['a', 'b']}
            path = manager.save_checkpoint('test', state)

            assert os.path.exists(path)

    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        from src.utils.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            state = {'processed': 10}
            manager.save_checkpoint('test', state)

            loaded = manager.load_checkpoint('latest')
            assert loaded['state'] == state

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        from src.utils.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            manager.save_checkpoint('task1', {})
            manager.save_checkpoint('task2', {})

            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 2

    def test_delete_checkpoint(self):
        """Test deleting checkpoint."""
        from src.utils.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            path = manager.save_checkpoint('test', {})
            assert os.path.exists(path)

            manager.delete_checkpoint(path)
            assert not os.path.exists(path)


class TestResumableTask:
    """Tests for resumable task."""

    def test_mark_processed(self):
        """Test marking items as processed."""
        from src.utils.checkpoint import ResumableTask, CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            task = ResumableTask('test', manager)

            task.mark_processed('item_001')
            task.mark_processed('item_002')

            assert task.processed_count == 2
            assert task.is_processed('item_001')
            assert not task.is_processed('item_003')

    def test_get_remaining_items(self):
        """Test getting remaining items."""
        from src.utils.checkpoint import ResumableTask, CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            task = ResumableTask('test', manager)

            task.mark_processed('item_001')

            all_items = ['item_001', 'item_002', 'item_003']
            remaining = task.get_remaining_items(all_items)

            assert 'item_001' not in remaining
            assert 'item_002' in remaining
            assert 'item_003' in remaining

    def test_resume_from_checkpoint(self):
        """Test resuming from checkpoint."""
        from src.utils.checkpoint import ResumableTask, CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            # Create and save task
            task1 = ResumableTask('test', manager)
            task1.mark_processed('item_001')
            task1.mark_processed('item_002')
            task1.update_statistics('total', 100)
            task1.save_checkpoint()

            # Resume in new task
            task2 = ResumableTask('test', manager)
            task2.resume_from_checkpoint('latest')

            assert task2.processed_count == 2
            assert task2.statistics['total'] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
