"""Google Gemini API provider."""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Any, List, Set

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
        self._api_keys = self._load_api_keys()
        self._current_key_index = 0
        # Tracks which key indices have hit their daily quota
        self._exhausted_keys: Set[int] = set()

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
        if not self._api_keys:
            return False

        try:
            import google.generativeai
            return True
        except ImportError:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate content using Gemini.

        Rotates through all API keys on quota/rate errors. When every key has
        hit its *daily* quota limit, sleeps until 00:00 AM for the reset and
        then retries automatically — allowing non-stop synthesis runs.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction

        Returns:
            GenerationResult with generated text
        """
        MAX_MIDNIGHT_WAITS = 2  # safety cap: don't loop past 2 daily resets
        midnight_waits = 0

        while True:
            attempts = max(len(self._api_keys), 1)
            last_error: Optional[Exception] = None
            all_daily_exhausted = False

            for _ in range(attempts):
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
                    error_msg = str(e)
                    error_lower = error_msg.lower()
                    last_error = e

                    if 'rate' in error_lower or 'quota' in error_lower or '429' in error_lower:
                        if self._is_daily_quota_error(error_msg):
                            # Mark this key as daily-exhausted
                            self._exhausted_keys.add(self._current_key_index)
                            self.logger.warning(
                                f"Gemini API key #{self._current_key_index + 1} "
                                f"hit daily quota ({len(self._exhausted_keys)}/{len(self._api_keys)} keys exhausted)"
                            )
                            # Try rotating to a key that still has quota
                            if self._rotate_to_available_key():
                                continue
                            # All keys are daily-exhausted — trigger midnight wait
                            all_daily_exhausted = True
                            break
                        else:
                            # Temporary per-minute/per-second rate limit — just rotate
                            if self._rotate_api_key():
                                continue
                            raise RateLimitError(f"Gemini rate limited: {e}")
                    elif 'auth' in error_lower or 'key' in error_lower or '401' in error_lower:
                        if self._rotate_api_key():
                            continue
                        raise AuthenticationError(f"Gemini auth failed: {e}")
                    else:
                        raise ProviderError(f"Gemini error: {e}")

            if all_daily_exhausted:
                if midnight_waits >= MAX_MIDNIGHT_WAITS:
                    raise RateLimitError(
                        f"Gemini daily quota exhausted on all {len(self._api_keys)} key(s) "
                        f"after {midnight_waits} midnight reset(s): {last_error}"
                    )
                self._wait_for_daily_reset()
                midnight_waits += 1
                # Outer while-loop continues: retry with fresh keys after reset
            else:
                # Inner loop exhausted normally (temporary rate limits, no daily exhaustion)
                raise RateLimitError(f"Gemini rate limited across all keys: {last_error}")

    def _load_api_keys(self) -> List[str]:
        """Load Gemini API keys from config or environment."""
        keys = []
        if self.api_key:
            keys.extend(self._split_keys(self.api_key))

        env_keys = os.getenv("GEMINI_API_KEYS", "")
        if env_keys:
            keys.extend(self._split_keys(env_keys))

        unique_keys = []
        for key in keys:
            if key and key not in unique_keys:
                unique_keys.append(key)
        if unique_keys:
            self.api_key = unique_keys[0]
        return unique_keys

    @staticmethod
    def _split_keys(value: str) -> List[str]:
        return [key.strip() for key in value.split(",") if key.strip()]

    def _rotate_api_key(self) -> bool:
        """Rotate to the next API key (used for temporary rate limits)."""
        if len(self._api_keys) <= 1:
            return False

        self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
        self.api_key = self._api_keys[self._current_key_index]
        self._client = None
        self._model = None
        self.logger.warning(f"Rotated Gemini API key to #{self._current_key_index + 1}")
        return True

    def _rotate_to_available_key(self) -> bool:
        """Rotate to the next key that has NOT hit its daily quota."""
        available = [i for i in range(len(self._api_keys)) if i not in self._exhausted_keys]
        if not available:
            return False

        # Pick the first available key that differs from the current one
        for i in available:
            if i != self._current_key_index:
                self._current_key_index = i
                self.api_key = self._api_keys[i]
                self._client = None
                self._model = None
                self.logger.info(f"Switched to non-exhausted Gemini API key #{i + 1}")
                return True

        return False  # Only the current (exhausted) key was in the list

    def _is_daily_quota_error(self, error_msg: str) -> bool:
        """Return True when the error indicates a *daily* quota exhaustion.

        Daily quota errors come from RESOURCE_EXHAUSTED / per-day limits.
        Per-minute / per-second rate limits are NOT daily exhaustion.
        """
        lower = error_msg.lower()
        daily_signals = [
            'resource_exhausted', 'quota exceeded', 'daily limit',
            'per day', 'daily quota', 'exceeded your current quota',
        ]
        rate_signals = [
            'per minute', 'per second', 'requests per', 'rate limit exceeded',
        ]
        is_daily = any(s in lower for s in daily_signals)
        is_rate = any(s in lower for s in rate_signals)
        return is_daily and not is_rate

    def _wait_for_daily_reset(self) -> None:
        """Sleep until 00:00 AM local time (Google's daily quota reset).

        Logs a countdown every 30 minutes. After waking, clears all
        exhausted-key flags so synthesis resumes with fresh daily limits.
        """
        now = datetime.now()
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        wait_seconds = (midnight - now).total_seconds()

        self.logger.warning(
            f"All {len(self._api_keys)} Gemini API key(s) have hit their daily quota limit. "
            f"Sleeping {wait_seconds / 3600:.1f}h until reset at "
            f"{midnight.strftime('%Y-%m-%d 00:00:00')} — synthesis will resume automatically."
        )

        log_interval = 1800  # log every 30 minutes
        slept = 0.0
        while slept < wait_seconds:
            chunk = min(log_interval, wait_seconds - slept)
            time.sleep(chunk)
            slept += chunk
            remaining = wait_seconds - slept
            if remaining > 60:
                self.logger.info(
                    f"Waiting for daily quota reset... {remaining / 3600:.1f}h remaining "
                    f"(resets at {midnight.strftime('%H:%M:%S')})"
                )

        # Clear state — all keys are fresh again after midnight
        self._exhausted_keys.clear()
        self._current_key_index = 0
        self.api_key = self._api_keys[0]
        self._client = None
        self._model = None
        self.logger.info(
            f"Daily quota reset complete. All {len(self._api_keys)} Gemini API key(s) "
            f"are available again. Resuming synthesis..."
        )

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

    from ..core.config import load_config

    config = load_config("config.yaml")
    gemini_config = config.get_provider_config("gemini")
    if not gemini_config:
        print("  ⚠ Gemini provider not configured in config.yaml")
        exit()

    provider = GeminiProvider(gemini_config)

    # Test 1: Check availability
    available = provider.is_available()
    print(f"  {'✓' if available else '✗'} Provider available: {available}")

    if not gemini_config.api_key:
        print("  ⚠ No API key - skipping live tests")
        print("  Set GEMINI_API_KEY or GEMINI_API_KEYS to run live tests")
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
