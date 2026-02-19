"""OpenRouter API provider."""

import time
import requests
from typing import Optional, Any

from .base import (
    BaseProvider, GenerationResult, FinishReason,
    ProviderError, RateLimitError, SafetyFilterError, AuthenticationError
)
from ..core.logger import get_logger


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: Any):
        """Initialize OpenRouter provider.

        Args:
            config: ProviderConfig for OpenRouter
        """
        super().__init__(config)
        self.logger = get_logger(__name__)

    def is_available(self) -> bool:
        """Check if OpenRouter provider is available."""
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate content using OpenRouter.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction

        Returns:
            GenerationResult with generated text
        """
        if not self.api_key:
            raise AuthenticationError("OpenRouter API key not configured")

        # Rate limiting
        time.sleep(self.rate_limit_delay)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "top_p": self.top_p
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/data-synthesizer",
            "X-Title": "Data Synthesizer"
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            # Handle HTTP errors
            if response.status_code == 429:
                raise RateLimitError("OpenRouter rate limited")
            elif response.status_code == 401:
                raise AuthenticationError("OpenRouter authentication failed")
            elif response.status_code != 200:
                raise ProviderError(f"OpenRouter error: {response.status_code} - {response.text}")

            data = response.json()

            # Check for error in response
            if "error" in data:
                error_msg = data["error"].get("message", str(data["error"]))
                raise ProviderError(f"OpenRouter error: {error_msg}")

            # Extract response
            choices = data.get("choices", [])
            if not choices:
                return GenerationResult(
                    text="",
                    finish_reason=FinishReason.ERROR,
                    provider=self.name,
                    model=self.model
                )

            choice = choices[0]
            text = choice.get("message", {}).get("content", "").strip()

            # Map finish reason
            finish_reason_str = choice.get("finish_reason", "stop")
            finish_reason = self._map_finish_reason(finish_reason_str)

            # Get token usage
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

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

        except (RateLimitError, AuthenticationError, SafetyFilterError):
            raise
        except requests.exceptions.Timeout:
            raise ProviderError("OpenRouter request timed out")
        except requests.exceptions.RequestException as e:
            raise ProviderError(f"OpenRouter network error: {e}")
        except Exception as e:
            raise ProviderError(f"OpenRouter error: {e}")

    def _map_finish_reason(self, reason: str) -> FinishReason:
        """Map OpenRouter finish reason to our enum.

        Args:
            reason: Finish reason string

        Returns:
            FinishReason enum value
        """
        reason = reason.lower()

        if reason in ["stop", "end_turn"]:
            return FinishReason.COMPLETE
        elif reason == "length":
            return FinishReason.MAX_TOKENS
        elif reason in ["content_filter", "safety"]:
            return FinishReason.SAFETY
        else:
            return FinishReason.ERROR


if __name__ == "__main__":
    print("=" * 60)
    print("OPENROUTER PROVIDER TEST")
    print("=" * 60)

    import os
    from ..core.config import ProviderConfig

    # Create test config
    config = ProviderConfig(
        name="openrouter",
        model="google/gemini-flash-1.5",
        temperature=0.7,
        max_output_tokens=1000,
        rate_limit_delay=1.0,
        api_key=os.getenv("OPENROUTER_API_KEY", "")
    )

    provider = OpenRouterProvider(config)

    # Test 1: Check availability
    available = provider.is_available()
    print(f"  {'✓' if available else '✗'} Provider available: {available}")

    if not config.api_key:
        print("  ⚠ No API key - skipping live tests")
        print("  Set OPENROUTER_API_KEY to run live tests")
    else:
        # Test 2: Simple generation
        try:
            result = provider.generate("Say 'Hello, World!' in exactly those words.")
            print(f"  ✓ Generation successful: '{result.text[:50]}...'")
            print(f"    Finish reason: {result.finish_reason}")
            print(f"    Tokens used: {result.tokens_used}")
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")

        # Test 3: With system prompt
        try:
            result = provider.generate(
                "What is 2+2?",
                system_prompt="You are a helpful math tutor. Be concise."
            )
            print(f"  ✓ System prompt generation: '{result.text[:50]}...'")
        except Exception as e:
            print(f"  ✗ System prompt test failed: {e}")

        # Test 4: Usage stats
        usage = provider.get_usage()
        print(f"  ✓ Usage stats: {usage['total_requests']} requests, {usage['total_tokens']} tokens")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
