"""Google Gemini API provider."""

import time
from typing import Optional, Any

from .base import (
    BaseProvider, GenerationResult, FinishReason,
    ProviderError, RateLimitError, SafetyFilterError, AuthenticationError
)
from ..core.logger import get_logger


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini API."""

    def __init__(self, config: Any):
        """Initialize Gemini provider.

        Args:
            config: ProviderConfig for Gemini
        """
        super().__init__(config)
        self.logger = get_logger(__name__)
        self._client = None
        self._model = None

    def _initialize_client(self) -> None:
        """Initialize Gemini client lazily."""
        if self._client is not None:
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai

            # Create model with generation config
            self._model = genai.GenerativeModel(
                self.model,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    top_p=0.8,
                    top_k=40
                )
            )
            self.logger.debug(f"Gemini client initialized with model {self.model}")

        except ImportError:
            raise ProviderError("google-generativeai package not installed")
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize Gemini: {e}")

    def is_available(self) -> bool:
        """Check if Gemini provider is available."""
        if not self.api_key:
            return False

        try:
            import google.generativeai
            return True
        except ImportError:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate content using Gemini.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction

        Returns:
            GenerationResult with generated text
        """
        self._initialize_client()

        # Rate limiting
        time.sleep(self.rate_limit_delay)

        try:
            # Use system instruction if provided
            if system_prompt:
                model = self._client.GenerativeModel(
                    self.model,
                    generation_config=self._client.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        top_p=0.8,
                        top_k=40
                    ),
                    system_instruction=system_prompt
                )
                response = model.generate_content(prompt)
            else:
                response = self._model.generate_content(prompt)

            # Check for valid response
            if not response.candidates:
                self.logger.warning("No candidates returned from Gemini")
                return GenerationResult(
                    text="",
                    finish_reason=FinishReason.ERROR,
                    provider=self.name,
                    model=self.model
                )

            candidate = response.candidates[0]

            # Map finish reason
            finish_reason = self._map_finish_reason(candidate.finish_reason)

            # Handle safety filter
            if finish_reason == FinishReason.SAFETY:
                raise SafetyFilterError("Content filtered by safety settings")

            # Extract text
            if not candidate.content or not candidate.content.parts:
                return GenerationResult(
                    text="",
                    finish_reason=FinishReason.ERROR,
                    provider=self.name,
                    model=self.model
                )

            text = candidate.content.parts[0].text.strip()

            # Get token usage if available
            tokens_used = 0
            if hasattr(response, 'usage_metadata'):
                tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)

            result = GenerationResult(
                text=text,
                finish_reason=finish_reason,
                tokens_used=tokens_used,
                provider=self.name,
                model=self.model
            )

            self._update_usage(result)
            return result

        except SafetyFilterError:
            raise
        except Exception as e:
            error_msg = str(e).lower()

            if 'rate' in error_msg or 'quota' in error_msg or '429' in error_msg:
                raise RateLimitError(f"Gemini rate limited: {e}")
            elif 'auth' in error_msg or 'key' in error_msg or '401' in error_msg:
                raise AuthenticationError(f"Gemini auth failed: {e}")
            else:
                raise ProviderError(f"Gemini error: {e}")

    def _map_finish_reason(self, reason) -> FinishReason:
        """Map Gemini finish reason to our enum.

        Args:
            reason: Gemini finish reason code

        Returns:
            FinishReason enum value
        """
        # Gemini finish reason codes
        # 1 = STOP (complete)
        # 2 = SAFETY
        # 3 = RECITATION
        # 4 = OTHER (error)
        # 5 = MAX_TOKENS

        if reason == 1:
            return FinishReason.COMPLETE
        elif reason == 2:
            return FinishReason.SAFETY
        elif reason == 3:
            return FinishReason.RECITATION
        elif reason == 5:
            return FinishReason.MAX_TOKENS
        else:
            return FinishReason.ERROR


if __name__ == "__main__":
    print("=" * 60)
    print("GEMINI PROVIDER TEST")
    print("=" * 60)

    import os
    from ..core.config import ProviderConfig

    # Create test config
    config = ProviderConfig(
        name="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        max_output_tokens=1000,
        rate_limit_delay=1.0,
        api_key=os.getenv("GEMINI_API_KEY", "")
    )

    provider = GeminiProvider(config)

    # Test 1: Check availability
    available = provider.is_available()
    print(f"  {'✓' if available else '✗'} Provider available: {available}")

    if not config.api_key:
        print("  ⚠ No API key - skipping live tests")
        print("  Set GEMINI_API_KEY to run live tests")
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
