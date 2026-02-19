"""OpenAI API provider."""

import time
from typing import Optional, Any

from .base import (
    BaseProvider, GenerationResult, FinishReason,
    ProviderError, RateLimitError, SafetyFilterError, AuthenticationError
)
from ..core.logger import get_logger


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI API."""

    def __init__(self, config: Any):
        """Initialize OpenAI provider.

        Args:
            config: ProviderConfig for OpenAI
        """
        super().__init__(config)
        self.logger = get_logger(__name__)
        self._client = None

    def _initialize_client(self) -> None:
        """Initialize OpenAI client lazily."""
        if self._client is not None:
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            self.logger.debug(f"OpenAI client initialized with model {self.model}")

        except ImportError:
            raise ProviderError("openai package not installed")
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize OpenAI: {e}")

    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        if not self.api_key:
            return False

        try:
            import openai
            return True
        except ImportError:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate content using OpenAI.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction

        Returns:
            GenerationResult with generated text
        """
        self._initialize_client()

        # Rate limiting
        time.sleep(self.rate_limit_delay)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=self.top_p
            )

            # Check for valid response
            if not response.choices:
                return GenerationResult(
                    text="",
                    finish_reason=FinishReason.ERROR,
                    provider=self.name,
                    model=self.model
                )

            choice = response.choices[0]
            text = choice.message.content.strip() if choice.message.content else ""

            # Map finish reason
            finish_reason = self._map_finish_reason(choice.finish_reason)

            # Get token usage
            tokens_used = 0
            prompt_tokens = 0
            completion_tokens = 0

            if response.usage:
                tokens_used = response.usage.total_tokens
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens

            result = GenerationResult(
                text=text,
                finish_reason=finish_reason,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider=self.name,
                model=self.model
            )

            self._update_usage(result)
            return result

        except Exception as e:
            error_msg = str(e).lower()

            if 'rate' in error_msg or '429' in error_msg:
                raise RateLimitError(f"OpenAI rate limited: {e}")
            elif 'auth' in error_msg or 'key' in error_msg or '401' in error_msg:
                raise AuthenticationError(f"OpenAI auth failed: {e}")
            elif 'content_filter' in error_msg or 'content_policy' in error_msg:
                raise SafetyFilterError(f"OpenAI content filtered: {e}")
            else:
                raise ProviderError(f"OpenAI error: {e}")

    def _map_finish_reason(self, reason: str) -> FinishReason:
        """Map OpenAI finish reason to our enum.

        Args:
            reason: Finish reason string

        Returns:
            FinishReason enum value
        """
        if reason == "stop":
            return FinishReason.COMPLETE
        elif reason == "length":
            return FinishReason.MAX_TOKENS
        elif reason == "content_filter":
            return FinishReason.SAFETY
        else:
            return FinishReason.ERROR


if __name__ == "__main__":
    print("=" * 60)
    print("OPENAI PROVIDER TEST")
    print("=" * 60)

    import os
    from ..core.config import ProviderConfig

    config = ProviderConfig(
        name="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        max_output_tokens=1000,
        rate_limit_delay=1.0,
        api_key=os.getenv("OPENAI_API_KEY", "")
    )

    provider = OpenAIProvider(config)

    available = provider.is_available()
    print(f"  {'✓' if available else '✗'} Provider available: {available}")

    if not config.api_key:
        print("  ⚠ No API key - skipping live tests")
        print("  Set OPENAI_API_KEY to run live tests")
    else:
        try:
            result = provider.generate("Say 'Hello' and nothing else.")
            print(f"  ✓ Generation: '{result.text[:50]}...'")
            print(f"    Tokens: {result.tokens_used}")
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
