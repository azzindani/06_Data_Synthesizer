"""Unit tests for configuration module."""

import pytest
import os

from src.core.config import (
    Config, ProviderConfig, SynthesisConfig, QualityConfig,
    RetryConfig, OutputConfig, load_config, _create_default_config, _parse_config
)


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_create_with_defaults(self):
        """Test creating provider config with default values."""
        config = ProviderConfig(name="test", model="test-model")
        assert config.name == "test"
        assert config.model == "test-model"
        assert config.temperature == 0.7
        assert config.max_output_tokens == 8000
        assert config.rate_limit_delay == 4.0

    def test_api_key_from_environment(self, monkeypatch):
        """Test loading API key from environment."""
        monkeypatch.setenv("TEST_API_KEY", "secret_key")
        config = ProviderConfig(name="test", model="test-model")
        assert config.api_key == "secret_key"

    def test_explicit_api_key(self):
        """Test explicit API key takes precedence."""
        config = ProviderConfig(name="test", model="test-model", api_key="explicit")
        assert config.api_key == "explicit"


class TestConfig:
    """Tests for main Config class."""

    def test_default_creation(self):
        """Test creating default config."""
        config = _create_default_config()
        assert config.app_name == "data-synthesizer"
        assert config.primary_provider == "gemini"
        assert "gemini" in config.providers
        assert "openrouter" in config.providers

    def test_get_provider_config(self, default_config):
        """Test getting provider configuration."""
        gemini = default_config.get_provider_config("gemini")
        assert gemini is not None
        assert gemini.name == "gemini"

        unknown = default_config.get_provider_config("unknown")
        assert unknown is None

    def test_get_active_providers(self, default_config):
        """Test getting active providers list."""
        providers = default_config.get_active_providers()
        assert "gemini" in providers
        assert providers[0] == default_config.primary_provider

    def test_validation_no_api_key(self, default_config):
        """Test validation fails without API key."""
        # Clear all API keys
        for provider in default_config.providers.values():
            provider.api_key = ""

        errors = default_config.validate()
        assert len(errors) > 0
        assert any("API key" in e for e in errors)

    def test_validation_missing_provider(self, default_config):
        """Test validation fails with missing provider."""
        default_config.primary_provider = "nonexistent"
        errors = default_config.validate()
        assert any("nonexistent" in e for e in errors)

    def test_validation_hf_output(self, default_config):
        """Test validation for HuggingFace output."""
        default_config.output.type = "huggingface"
        default_config.output.repository = ""

        errors = default_config.validate()
        assert any("repository" in e for e in errors)


class TestLoadConfig:
    """Tests for config loading functions."""

    def test_load_default_on_missing_file(self):
        """Test loading default config when file doesn't exist."""
        config = load_config("nonexistent.yaml")
        assert config is not None
        assert config.app_name == "data-synthesizer"

    def test_parse_config_basic(self):
        """Test parsing basic config dictionary."""
        data = {
            'app': {
                'name': 'test-app',
                'mode': 'development'
            },
            'synthesis': {
                'type': 'deep_thinking',
                'num_variants': 3
            }
        }

        config = _parse_config(data)
        assert config.app_name == "test-app"
        assert config.mode == "development"
        assert config.synthesis.type == "deep_thinking"
        assert config.synthesis.num_variants == 3

    def test_parse_config_with_providers(self):
        """Test parsing config with provider settings."""
        data = {
            'providers': {
                'primary': 'openrouter',
                'fallback': ['gemini'],
                'openrouter': {
                    'model': 'custom-model',
                    'temperature': 0.9
                },
                'gemini': {
                    'model': 'gemini-pro'
                }
            }
        }

        config = _parse_config(data)
        assert config.primary_provider == "openrouter"
        assert "openrouter" in config.providers
        assert config.providers['openrouter'].model == "custom-model"
        assert config.providers['openrouter'].temperature == 0.9


class TestSubConfigs:
    """Tests for sub-configuration classes."""

    def test_synthesis_config_defaults(self):
        """Test SynthesisConfig default values."""
        config = SynthesisConfig()
        assert config.type == "qa"
        assert config.batch_size == 5
        assert config.num_variants == 5

    def test_quality_config_defaults(self):
        """Test QualityConfig default values."""
        config = QualityConfig()
        assert config.min_answer_length == 100
        assert config.target_language == "id"

    def test_retry_config_defaults(self):
        """Test RetryConfig default values."""
        config = RetryConfig()
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.exponential_base == 2.0

    def test_output_config_defaults(self):
        """Test OutputConfig default values."""
        config = OutputConfig()
        assert config.type == "huggingface"
        assert config.chunk_format == "parquet"
