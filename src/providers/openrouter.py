"""OpenRouter API provider.

Mirrors the Gemini provider's multi-key + interruptible-sleep pattern so that
OpenRouter can also run as an unstoppable cron-style backend: rotate keys on
per-minute 429s, and wait on the server-supplied ``Retry-After`` when every
key is throttled. The wait wakes immediately on SIGTERM via the shared
``core.shutdown`` event.

Keys can come from either:
  - ``OPENROUTER_API_KEY``        — single, or comma-separated list
  - ``OPENROUTER_API_KEYS``       — comma-separated list (preferred for many)
"""

import os
import time
import requests
from typing import Optional, Any, List

from .base import (
    BaseProvider, GenerationResult, FinishReason,
    ProviderError, RateLimitError, SafetyFilterError, AuthenticationError
)
from ..core.logger import get_logger
from ..core import shutdown
from ..utils.retry import interruptible_transient_retry, is_transient_error


# Cap on Retry-After-driven sleeps when every key is throttled at once.
# Anything above this likely indicates daily credit exhaustion rather than
# a minute-window throttle — better to surface as an error so the factory
# can fail over to another provider.
_MAX_RETRY_AFTER_SECONDS = 1800  # 30 minutes


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: Any):
        super().__init__(config)
        self.logger = get_logger(__name__)
        self._api_keys = self._load_api_keys()
        self._current_key_index = 0

    def is_available(self) -> bool:
        return bool(self._api_keys)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate content using OpenRouter with multi-key rotation."""
        if not self._api_keys:
            raise AuthenticationError("OpenRouter API key not configured")

        attempts = max(len(self._api_keys), 1)
        last_error: Optional[Exception] = None
        retry_after_seconds: Optional[float] = None

        for _ in range(attempts):
            if shutdown.is_requested():
                raise ProviderError("Shutdown requested during OpenRouter generate()")

            time.sleep(self.rate_limit_delay)

            try:
                return interruptible_transient_retry(
                    lambda: self._do_request(prompt, system_prompt),
                    max_attempts=3,
                    base_delay=2.0,
                    classifier=_openrouter_is_transient,
                    logger_name=__name__,
                )
            except RateLimitError as e:
                last_error = e
                retry_after_seconds = getattr(e, "retry_after", None)
                if self._rotate_api_key():
                    continue
                # No more keys to try
                break
            except AuthenticationError:
                if self._rotate_api_key():
                    continue
                raise
            # ProviderError / SafetyFilterError / others bubble up — the
            # factory will decide whether to fail over.

        # All keys throttled — wait if the server told us how long, otherwise
        # raise so the factory falls back to another provider.
        if retry_after_seconds is not None and retry_after_seconds <= _MAX_RETRY_AFTER_SECONDS:
            self.logger.warning(
                f"All {len(self._api_keys)} OpenRouter key(s) throttled. "
                f"Sleeping {retry_after_seconds:.0f}s per Retry-After."
            )
            completed = shutdown.interruptible_sleep(retry_after_seconds)
            if not completed:
                raise ProviderError("Shutdown requested during OpenRouter Retry-After wait")
            # Try once more after waking
            try:
                return self._do_request(prompt, system_prompt)
            except RateLimitError as e:
                last_error = e

        raise RateLimitError(f"OpenRouter rate limited across all keys: {last_error}")

    def _do_request(self, prompt: str, system_prompt: Optional[str]) -> GenerationResult:
        """Single HTTP request — raises typed exceptions on failure."""
        messages: List[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "top_p": self.top_p,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/data-synthesizer",
            "X-Title": "Data Synthesizer",
        }

        try:
            response = requests.post(self.API_URL, headers=headers, json=payload, timeout=120)
        except requests.exceptions.Timeout:
            raise ProviderError("OpenRouter request timed out")
        except requests.exceptions.RequestException as e:
            raise ProviderError(f"OpenRouter network error: {e}")

        if response.status_code == 429:
            err = RateLimitError("OpenRouter rate limited")
            # Stash Retry-After (seconds) for the caller's wait logic.
            err.retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            raise err
        if response.status_code == 401:
            raise AuthenticationError("OpenRouter authentication failed")
        if response.status_code != 200:
            raise ProviderError(f"OpenRouter error: {response.status_code} - {response.text[:300]}")

        data = response.json()
        if "error" in data:
            error_msg = data["error"].get("message", str(data["error"]))
            raise ProviderError(f"OpenRouter error: {error_msg}")

        choices = data.get("choices", [])
        if not choices:
            return GenerationResult(text="", finish_reason=FinishReason.ERROR,
                                    provider=self.name, model=self.model)

        choice = choices[0]
        text = choice.get("message", {}).get("content", "").strip()
        finish_reason = self._map_finish_reason(choice.get("finish_reason", "stop"))

        usage = data.get("usage", {})
        result = GenerationResult(
            text=text,
            finish_reason=finish_reason,
            tokens_used=usage.get("total_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            provider=self.name,
            model=self.model,
        )
        self._update_usage(result)
        return result

    # ── multi-key plumbing ──────────────────────────────────────────────────

    def _load_api_keys(self) -> List[str]:
        keys: List[str] = []
        if self.api_key:
            keys.extend(self._split(self.api_key))
        env_keys = os.getenv("OPENROUTER_API_KEYS", "")
        if env_keys:
            keys.extend(self._split(env_keys))

        unique: List[str] = []
        for k in keys:
            if k and k not in unique:
                unique.append(k)
        if unique:
            self.api_key = unique[0]
        return unique

    @staticmethod
    def _split(value: str) -> List[str]:
        return [k.strip() for k in value.split(",") if k.strip()]

    def _rotate_api_key(self) -> bool:
        if len(self._api_keys) <= 1:
            return False
        self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
        self.api_key = self._api_keys[self._current_key_index]
        self.logger.warning(f"Rotated OpenRouter API key to #{self._current_key_index + 1}")
        return True

    def attach_progress_manager(self, progress_manager: Any) -> None:
        """Optional hook — symmetric with GeminiProvider so the synthesizer
        can wire both providers without special-casing."""
        self._progress_manager = progress_manager

    # ── helpers ─────────────────────────────────────────────────────────────

    def _map_finish_reason(self, reason: str) -> FinishReason:
        reason = (reason or "").lower()
        if reason in ("stop", "end_turn"):
            return FinishReason.COMPLETE
        if reason == "length":
            return FinishReason.MAX_TOKENS
        if reason in ("content_filter", "safety"):
            return FinishReason.SAFETY
        return FinishReason.ERROR


def _openrouter_is_transient(err: Exception) -> bool:
    """Retry-classifier specific to OpenRouter: never retry typed errors that
    are meant to drive key-rotation or failover."""
    if isinstance(err, (RateLimitError, AuthenticationError, SafetyFilterError)):
        return False
    return is_transient_error(err)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a Retry-After header value. Returns seconds, or None when absent.

    Per RFC 9110, the value can be either an integer count of seconds or an
    HTTP-date. We handle the seconds form (the only form OpenRouter uses in
    practice) and treat anything else as "unknown — caller decides".
    """
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("OPENROUTER PROVIDER TEST")
    print("=" * 60)

    from ..core.config import ProviderConfig

    config = ProviderConfig(
        name="openrouter",
        model="google/gemini-flash-1.5",
        temperature=0.7,
        max_output_tokens=1000,
        rate_limit_delay=1.0,
        api_key=os.getenv("OPENROUTER_API_KEY", "")
    )

    provider = OpenRouterProvider(config)
    print(f"  ✓ Provider available: {provider.is_available()}")
    print(f"  ✓ Loaded {len(provider._api_keys)} key(s)")

    if not provider._api_keys:
        print("  ⚠ No keys — set OPENROUTER_API_KEY or OPENROUTER_API_KEYS to run live tests")
    else:
        try:
            result = provider.generate("Say 'Hello, World!' in exactly those words.")
            print(f"  ✓ Generation: '{result.text[:50]}...'")
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
