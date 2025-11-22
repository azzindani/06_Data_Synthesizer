"""Unit tests for Phase 3 components."""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_initialization(self):
        """Test provider initialization."""
        from src.providers.openai_provider import OpenAIProvider

        config = Mock()
        config.name = "openai"
        config.api_key = "test-key"
        config.model = "gpt-4"
        config.temperature = 0.7
        config.max_output_tokens = 2048
        config.rate_limit_delay = 1

        provider = OpenAIProvider(config)
        assert provider.name == "openai"
        assert provider.model == "gpt-4"

    def test_is_available(self):
        """Test availability check."""
        from src.providers.openai_provider import OpenAIProvider

        config = Mock()
        config.name = "openai"
        config.api_key = "test-key"
        config.model = "gpt-4"
        config.temperature = 0.7
        config.max_output_tokens = 2048
        config.rate_limit_delay = 1

        provider = OpenAIProvider(config)
        # Will return False because openai package is not installed
        result = provider.is_available()
        assert isinstance(result, bool)

    @patch('src.providers.openai_provider.OpenAI')
    def test_generate(self, mock_openai_class):
        """Test text generation."""
        from src.providers.openai_provider import OpenAIProvider
        from src.providers.base import FinishReason

        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client.chat.completions.create.return_value = mock_response

        config = Mock()
        config.name = "openai"
        config.api_key = "test-key"
        config.model = "gpt-4"
        config.temperature = 0.7
        config.max_output_tokens = 2048
        config.rate_limit_delay = 0

        provider = OpenAIProvider(config)
        result = provider.generate("Test prompt")

        assert result.success is True
        assert result.text == "Test response"
        assert result.finish_reason == FinishReason.COMPLETE


class TestCostTracker:
    """Tests for cost tracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        from src.utils.cost_tracker import CostTracker

        tracker = CostTracker()
        assert len(tracker.records) == 0

    def test_record_usage(self):
        """Test recording API usage."""
        from src.utils.cost_tracker import CostTracker

        tracker = CostTracker()
        cost = tracker.record_usage("gemini", "gemini-1.5-flash", 1000, 500)

        assert cost > 0
        assert len(tracker.records) == 1

    def test_get_summary(self):
        """Test getting usage summary."""
        from src.utils.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record_usage("gemini", "gemini-1.5-flash", 1000, 500)
        tracker.record_usage("openai", "gpt-4", 500, 200)

        summary = tracker.get_summary()

        assert summary['total_requests'] == 2
        assert summary['total_cost_usd'] > 0
        assert 'gemini' in summary['by_provider']
        assert 'openai' in summary['by_provider']

    def test_estimate_remaining_cost(self):
        """Test cost estimation."""
        from src.utils.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record_usage("gemini", "gemini-1.5-flash", 1000, 500)

        estimate = tracker.estimate_remaining_cost(10)
        assert estimate > 0

    def test_get_recent_records(self):
        """Test getting recent records."""
        from src.utils.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record_usage("gemini", "gemini-1.5-flash", 1000, 500)

        records = tracker.get_recent_records(10)
        assert len(records) == 1
        assert records[0]['provider'] == 'gemini'


class TestNotificationManager:
    """Tests for notification manager."""

    def test_initialization(self):
        """Test manager initialization."""
        from src.utils.notifications import NotificationManager

        manager = NotificationManager()
        assert manager.slack_webhook == os.getenv('SLACK_WEBHOOK_URL', '')

    def test_no_config(self):
        """Test with no configuration."""
        from src.utils.notifications import NotificationManager

        manager = NotificationManager()

        # Should return False when not configured
        result = manager.send_slack("Test")
        assert result is False

        result = manager.send_email("Subject", "Body")
        assert result is False

    @patch('requests.post')
    def test_send_slack(self, mock_post):
        """Test Slack notification."""
        from src.utils.notifications import NotificationManager

        mock_post.return_value.status_code = 200

        manager = NotificationManager()
        manager.slack_webhook = "https://hooks.slack.com/test"

        result = manager.send_slack("Test message", "Title")
        assert result is True
        mock_post.assert_called_once()

    def test_notify_completion(self):
        """Test completion notification format."""
        from src.utils.notifications import NotificationManager

        manager = NotificationManager()

        stats = {
            'total_processed': 100,
            'total_generated': 500,
            'request_count': 100,
            'duration': '2h 30m',
            'total_cost': 0.0234,
            'output_location': 'test/output'
        }

        # Will return False as not configured, but shouldn't error
        results = manager.notify_completion(stats)
        assert 'slack' in results
        assert 'email' in results


class TestLocalOutputHandler:
    """Tests for local output handler."""

    def test_initialization(self):
        """Test handler initialization."""
        from src.utils.output_handler import LocalOutputHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalOutputHandler(tmpdir)
            assert handler.base_path.exists()

    def test_save_batch(self):
        """Test batch saving."""
        from src.utils.output_handler import LocalOutputHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalOutputHandler(tmpdir)

            data = [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ]

            path = handler.save_batch(data, "test")
            assert os.path.exists(path)

            # Verify content
            with open(path) as f:
                lines = f.readlines()
                assert len(lines) == 2

    def test_save_json(self):
        """Test JSON saving."""
        from src.utils.output_handler import LocalOutputHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalOutputHandler(tmpdir)

            data = {"test": "value", "count": 42}
            path = handler.save_json(data, "config")

            assert os.path.exists(path)

            with open(path) as f:
                loaded = json.load(f)
                assert loaded == data

    def test_checkpoint(self):
        """Test checkpoint save/load."""
        from src.utils.output_handler import LocalOutputHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalOutputHandler(tmpdir)

            state = {"processed": 50, "errors": []}
            handler.save_checkpoint(state)

            loaded = handler.load_checkpoint()
            assert loaded == state

    def test_get_item_count(self):
        """Test item counting."""
        from src.utils.output_handler import LocalOutputHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = LocalOutputHandler(tmpdir)

            data = [{"id": i} for i in range(5)]
            handler.save_batch(data, "test")

            count = handler.get_item_count("test")
            assert count == 5


class TestDomainConfigs:
    """Tests for domain configuration templates."""

    def test_indonesian_legal_config(self):
        """Test Indonesian legal config loading."""
        import yaml

        config_path = "configs/indonesian_legal.yaml"
        assert os.path.exists(config_path)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config['domain']['name'] == "Indonesian Legal System"
        assert config['domain']['language'] == "id"
        assert len(config['topics']) > 0
        assert 'system_prompt' in config
        assert 'prompt_template' in config

    def test_contract_generation_config(self):
        """Test contract generation config loading."""
        import yaml

        config_path = "configs/contract_generation.yaml"
        assert os.path.exists(config_path)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config['domain']['name'] == "Legal Contract Corpus"
        assert 'taxonomy' in config
        assert len(config['taxonomy']['document_types']) > 0
        assert len(config['taxonomy']['industries']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
