"""Base provider interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is hit."""
    pass


class SafetyFilterError(ProviderError):
    """Raised when content is filtered by safety."""
    pass


class AuthenticationError(ProviderError):
    """Raised when authentication fails."""
    pass


class FinishReason(Enum):
    """Reasons for generation completion."""
    COMPLETE = "complete"
    MAX_TOKENS = "max_tokens"
    SAFETY = "safety"
    RECITATION = "recitation"
    ERROR = "error"


@dataclass
class GenerationResult:
    """Result from content generation."""
    text: str
    finish_reason: FinishReason
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""

    @property
    def success(self) -> bool:
        """Check if generation was successful."""
        return self.finish_reason in [FinishReason.COMPLETE, FinishReason.MAX_TOKENS]


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Any):
        """Initialize provider with configuration.

        Args:
            config: ProviderConfig instance
        """
        self.config = config
        self.name = config.name
        self.model = config.model
        self.temperature = config.temperature
        self.max_output_tokens = config.max_output_tokens
        self.rate_limit_delay = config.rate_limit_delay
        self.api_key = config.api_key

        # Usage tracking
        self._total_tokens = 0
        self._total_requests = 0
        self._errors = 0

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate content from prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RateLimitError: When rate limited
            SafetyFilterError: When content filtered
            AuthenticationError: When auth fails
            ProviderError: For other errors
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured.

        Returns:
            True if provider can be used
        """
        pass

    def get_usage(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with usage stats
        """
        return {
            'provider': self.name,
            'model': self.model,
            'total_tokens': self._total_tokens,
            'total_requests': self._total_requests,
            'errors': self._errors
        }

    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self._total_tokens = 0
        self._total_requests = 0
        self._errors = 0

    def _update_usage(self, result: GenerationResult) -> None:
        """Update usage tracking from result.

        Args:
            result: Generation result
        """
        self._total_tokens += result.tokens_used
        self._total_requests += 1
        if not result.success:
            self._errors += 1


if __name__ == "__main__":
    print("=" * 60)
    print("BASE PROVIDER TEST")
    print("=" * 60)

    # Test 1: GenerationResult
    result = GenerationResult(
        text="Generated content",
        finish_reason=FinishReason.COMPLETE,
        tokens_used=100,
        provider="test"
    )
    assert result.success == True
    print(f"  ✓ GenerationResult created: success={result.success}")

    # Test 2: Failed result
    failed = GenerationResult(
        text="",
        finish_reason=FinishReason.SAFETY,
        provider="test"
    )
    assert failed.success == False
    print(f"  ✓ Failed result: success={failed.success}")

    # Test 3: FinishReason enum
    assert FinishReason.COMPLETE.value == "complete"
    print("  ✓ FinishReason enum works")

    # Test 4: Exception types
    try:
        raise RateLimitError("Too many requests")
    except ProviderError as e:
        print(f"  ✓ RateLimitError is ProviderError: {e}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
