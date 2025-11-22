"""Unit tests for provider system."""

import pytest

from src.providers.base import (
    BaseProvider, GenerationResult, FinishReason,
    ProviderError, RateLimitError, SafetyFilterError, AuthenticationError
)
from src.core.config import ProviderConfig


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_success_complete(self):
        """Test success property for complete finish."""
        result = GenerationResult(
            text="test",
            finish_reason=FinishReason.COMPLETE
        )
        assert result.success == True

    def test_success_max_tokens(self):
        """Test success property for max tokens."""
        result = GenerationResult(
            text="test",
            finish_reason=FinishReason.MAX_TOKENS
        )
        assert result.success == True

    def test_failure_safety(self):
        """Test failure for safety filter."""
        result = GenerationResult(
            text="",
            finish_reason=FinishReason.SAFETY
        )
        assert result.success == False

    def test_failure_error(self):
        """Test failure for error."""
        result = GenerationResult(
            text="",
            finish_reason=FinishReason.ERROR
        )
        assert result.success == False


class TestFinishReason:
    """Tests for FinishReason enum."""

    def test_all_values_exist(self):
        """Test all expected values exist."""
        assert FinishReason.COMPLETE.value == "complete"
        assert FinishReason.MAX_TOKENS.value == "max_tokens"
        assert FinishReason.SAFETY.value == "safety"
        assert FinishReason.RECITATION.value == "recitation"
        assert FinishReason.ERROR.value == "error"


class TestProviderExceptions:
    """Tests for provider exceptions."""

    def test_rate_limit_is_provider_error(self):
        """Test RateLimitError inherits from ProviderError."""
        with pytest.raises(ProviderError):
            raise RateLimitError("Rate limited")

    def test_safety_filter_is_provider_error(self):
        """Test SafetyFilterError inherits from ProviderError."""
        with pytest.raises(ProviderError):
            raise SafetyFilterError("Filtered")

    def test_auth_error_is_provider_error(self):
        """Test AuthenticationError inherits from ProviderError."""
        with pytest.raises(ProviderError):
            raise AuthenticationError("Auth failed")


class TestGeminiProvider:
    """Tests for Gemini provider."""

    def test_is_available_without_key(self):
        """Test is_available returns False without API key."""
        config = ProviderConfig(name="gemini", model="gemini-pro", api_key="")

        from src.providers.gemini import GeminiProvider
        provider = GeminiProvider(config)

        assert provider.is_available() == False

    def test_is_available_with_key(self):
        """Test is_available with API key."""
        config = ProviderConfig(name="gemini", model="gemini-pro", api_key="test_key")

        from src.providers.gemini import GeminiProvider
        provider = GeminiProvider(config)

        # Will be True if google-generativeai is installed
        # Just test it doesn't crash
        _ = provider.is_available()

    def test_finish_reason_mapping(self):
        """Test Gemini finish reason mapping."""
        config = ProviderConfig(name="gemini", model="gemini-pro", api_key="test")

        from src.providers.gemini import GeminiProvider
        provider = GeminiProvider(config)

        assert provider._map_finish_reason(1) == FinishReason.COMPLETE
        assert provider._map_finish_reason(2) == FinishReason.SAFETY
        assert provider._map_finish_reason(3) == FinishReason.RECITATION
        assert provider._map_finish_reason(5) == FinishReason.MAX_TOKENS
        assert provider._map_finish_reason(99) == FinishReason.ERROR


class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""

    def test_is_available_without_key(self):
        """Test is_available returns False without API key."""
        config = ProviderConfig(name="openrouter", model="test", api_key="")

        from src.providers.openrouter import OpenRouterProvider
        provider = OpenRouterProvider(config)

        assert provider.is_available() == False

    def test_is_available_with_key(self):
        """Test is_available returns True with API key."""
        config = ProviderConfig(name="openrouter", model="test", api_key="test_key")

        from src.providers.openrouter import OpenRouterProvider
        provider = OpenRouterProvider(config)

        assert provider.is_available() == True

    def test_finish_reason_mapping(self):
        """Test OpenRouter finish reason mapping."""
        config = ProviderConfig(name="openrouter", model="test", api_key="test")

        from src.providers.openrouter import OpenRouterProvider
        provider = OpenRouterProvider(config)

        assert provider._map_finish_reason("stop") == FinishReason.COMPLETE
        assert provider._map_finish_reason("length") == FinishReason.MAX_TOKENS
        assert provider._map_finish_reason("content_filter") == FinishReason.SAFETY


class TestProviderFactory:
    """Tests for provider factory."""

    def test_create_with_gemini(self, config_with_keys):
        """Test creating factory with Gemini provider."""
        from src.providers.factory import ProviderFactory

        factory = ProviderFactory(config_with_keys)
        assert factory.current_provider_name == "gemini"

    def test_list_providers(self, config_with_keys):
        """Test listing available providers."""
        from src.providers.factory import ProviderFactory

        factory = ProviderFactory(config_with_keys)
        providers = factory.list_providers()

        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_get_usage(self, config_with_keys):
        """Test getting usage statistics."""
        from src.providers.factory import ProviderFactory

        factory = ProviderFactory(config_with_keys)
        usage = factory.get_usage()

        assert 'total_tokens' in usage
        assert 'total_requests' in usage
        assert 'by_provider' in usage
