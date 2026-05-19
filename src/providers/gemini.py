"""Google Gemini API provider."""

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, List, Set

try:
    from zoneinfo import ZoneInfo
    _PACIFIC = ZoneInfo("America/Los_Angeles")
except Exception:  # pragma: no cover — zoneinfo missing or tz data absent
    _PACIFIC = None

from .base import (
    BaseProvider, GenerationResult, FinishReason,
    ProviderError, RateLimitError, SafetyFilterError, AuthenticationError
)
from ..core.logger import get_logger
from ..core import shutdown
from ..utils.retry import interruptible_transient_retry, is_transient_error


def _next_quota_reset(now: Optional[datetime] = None,
                      buffer_minutes: int = 5) -> datetime:
    """Return the next Gemini daily-quota reset moment as an aware datetime.

    Free-tier per-day quotas reset on Pacific midnight (US/Pacific). We add a
    small ``buffer_minutes`` past midnight to avoid clock-skew races with
    Google's quota service. When IANA tz data is unavailable we fall back to
    "now + 24h", which always spans a reset regardless of timezone.
    """
    if _PACIFIC is None:
        return (now or datetime.now(timezone.utc)) + timedelta(hours=24, minutes=buffer_minutes)

    now_pt = (now.astimezone(_PACIFIC) if now else datetime.now(_PACIFIC))
    tomorrow_pt = (now_pt + timedelta(days=1)).replace(
        hour=0, minute=buffer_minutes, second=0, microsecond=0
    )
    return tomorrow_pt


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
                    top_p=self.top_p,
                    top_k=self.top_k
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
        hit its *daily* quota limit, sleeps until the next Gemini quota reset
        (Pacific midnight) and resumes automatically — repeats indefinitely so
        the synthesizer can run as an unstoppable cron-style job. Stops only
        when ``shutdown.is_requested()`` is true.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction

        Returns:
            GenerationResult with generated text
        """
        while True:
            if shutdown.is_requested():
                raise ProviderError("Shutdown requested during Gemini generate()")
            attempts = max(len(self._api_keys), 1)
            last_error: Optional[Exception] = None
            all_daily_exhausted = False

            for _ in range(attempts):
                self._initialize_client()

                # Rate limiting
                time.sleep(self.rate_limit_delay)

                try:
                    # Inner call is wrapped in transient retry so a 503 /
                    # network blip doesn't burn a quota or drop the item.
                    return interruptible_transient_retry(
                        lambda: self._do_single_request(prompt, system_prompt),
                        max_attempts=3,
                        base_delay=2.0,
                        classifier=_gemini_is_transient,
                        logger_name=__name__,
                    )

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
                # No cap — keep waiting through reset cycles until either a
                # generate() succeeds or shutdown is requested. The wait is
                # interruptible (SIGTERM wakes immediately).
                if not self._wait_for_daily_reset():
                    raise ProviderError("Shutdown requested during daily-quota wait")
                # Loop continues: retry with fresh keys after reset.
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

    def _wait_for_daily_reset(self) -> bool:
        """Sleep until the next Gemini daily-quota reset (Pacific midnight).

        Logs a countdown every 30 minutes. The sleep is interruptible — a
        SIGTERM during the wait wakes immediately. After waking on a real
        reset, clears all exhausted-key flags so synthesis resumes with
        fresh daily limits.

        Returns:
            True if the wait completed and quotas were reset; False if the
            wait was interrupted by a shutdown request.
        """
        reset_at = _next_quota_reset()
        # Compare in the same tz to get a correct delta.
        now_utc = datetime.now(timezone.utc)
        wait_seconds = (reset_at - now_utc).total_seconds()
        if wait_seconds <= 0:
            # Already past the next reset window — clear and proceed.
            self._reset_exhausted_state()
            return True

        reset_local_str = reset_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
        self.logger.warning(
            f"All {len(self._api_keys)} Gemini API key(s) have hit their daily quota. "
            f"Sleeping {wait_seconds / 3600:.1f}h until reset at {reset_local_str}. "
            f"Synthesis will resume automatically — SIGTERM to stop now."
        )

        self._publish_waiting_state(reset_at)

        def _heartbeat(remaining: float) -> None:
            if remaining > 60:
                self.logger.info(
                    f"Waiting for Gemini daily quota reset... "
                    f"{remaining / 3600:.1f}h remaining"
                )

        completed = shutdown.interruptible_sleep(
            wait_seconds,
            log_every=1800,
            log_message_fn=_heartbeat,
        )

        self._publish_waiting_state(None)

        if not completed:
            self.logger.info("Daily-quota wait interrupted by shutdown request")
            return False

        self._reset_exhausted_state()
        self.logger.info(
            f"Daily quota reset complete. All {len(self._api_keys)} Gemini API key(s) "
            f"are available again. Resuming synthesis..."
        )
        return True

    def _reset_exhausted_state(self) -> None:
        """Wipe per-key daily-exhaustion tracking and reset to first key."""
        self._exhausted_keys.clear()
        self._current_key_index = 0
        if self._api_keys:
            self.api_key = self._api_keys[0]
        self._client = None
        self._model = None

    def _publish_waiting_state(self, reset_at: Optional[datetime]) -> None:
        """Best-effort: tell the ProgressManager we're waiting for quota reset.

        Decoupled via duck-typing so the provider has no hard dep on
        ProgressManager — anything with ``set_waiting_state`` works.
        """
        progress = getattr(self, "_progress_manager", None)
        if progress is None or not hasattr(progress, "set_waiting_state"):
            return
        try:
            if reset_at is None:
                progress.set_waiting_state(None)
            else:
                progress.set_waiting_state(
                    reason="gemini_daily_quota",
                    until_iso=reset_at.astimezone(timezone.utc).isoformat(),
                    exhausted_keys=len(self._exhausted_keys),
                    total_keys=len(self._api_keys),
                )
        except Exception as e:
            self.logger.debug(f"Could not publish waiting state: {e}")

    def attach_progress_manager(self, progress_manager: Any) -> None:
        """Wire a ProgressManager in so quota-wait state is surfaced.

        Optional: providers work fine without it; the wait is just invisible
        to /progress when not attached.
        """
        self._progress_manager = progress_manager

    def _do_single_request(self, prompt: str, system_prompt: Optional[str]) -> GenerationResult:
        """One request to Gemini — raises typed errors. Wrapped by transient retry."""
        if system_prompt:
            model = self._client.GenerativeModel(
                self.model,
                generation_config=self._client.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    top_p=self.top_p,
                    top_k=self.top_k,
                ),
                system_instruction=system_prompt,
            )
            response = model.generate_content(prompt)
        else:
            response = self._model.generate_content(prompt)

        if not response.candidates:
            self.logger.warning("No candidates returned from Gemini")
            return GenerationResult(
                text="", finish_reason=FinishReason.ERROR,
                provider=self.name, model=self.model,
            )

        candidate = response.candidates[0]
        finish_reason = self._map_finish_reason(candidate.finish_reason)

        if finish_reason == FinishReason.SAFETY:
            raise SafetyFilterError("Content filtered by safety settings")

        if not candidate.content or not candidate.content.parts:
            return GenerationResult(
                text="", finish_reason=FinishReason.ERROR,
                provider=self.name, model=self.model,
            )

        text = candidate.content.parts[0].text.strip()
        tokens_used = 0
        if hasattr(response, "usage_metadata"):
            tokens_used = getattr(response.usage_metadata, "total_token_count", 0)

        result = GenerationResult(
            text=text, finish_reason=finish_reason,
            tokens_used=tokens_used, provider=self.name, model=self.model,
        )
        self._update_usage(result)
        return result

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


def _gemini_is_transient(err: Exception) -> bool:
    """Retry-classifier specific to Gemini: skip retries on errors that need
    key rotation or daily-quota handling instead — the outer loop does that."""
    if isinstance(err, (SafetyFilterError, RateLimitError, AuthenticationError)):
        return False
    msg = str(err).lower()
    # Don't retry quota / rate / auth in the inner transient loop — those are
    # owned by the per-key rotation logic above. Note we deliberately match
    # phrases that appear in the SDK's error text, not exception types,
    # because the SDK wraps things under a generic Exception.
    if "rate" in msg or "quota" in msg or "429" in msg:
        return False
    if "auth" in msg or "401" in msg:
        return False
    return is_transient_error(err)


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
