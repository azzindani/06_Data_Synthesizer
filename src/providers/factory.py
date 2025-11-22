"""Provider factory with automatic failover."""

from typing import List, Optional, Any

from .base import (
    BaseProvider, GenerationResult,
    ProviderError, RateLimitError, AuthenticationError
)
from .gemini import GeminiProvider
from .openrouter import OpenRouterProvider
from .openai_provider import OpenAIProvider
from ..core.logger import get_logger


# Provider registry
PROVIDER_CLASSES = {
    'gemini': GeminiProvider,
    'openrouter': OpenRouterProvider,
    'openai': OpenAIProvider,
}


class ProviderFactory:
    """Factory for creating providers with failover support."""

    def __init__(self, config: Any):
        """Initialize factory with configuration.

        Args:
            config: Main Config object
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize providers
        self._providers = {}
        self._provider_order = []
        self._current_index = 0

        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        # Get provider order from config
        provider_names = self.config.get_active_providers()

        for name in provider_names:
            if name not in PROVIDER_CLASSES:
                self.logger.warning(f"Unknown provider: {name}")
                continue

            provider_config = self.config.get_provider_config(name)
            if not provider_config:
                self.logger.warning(f"No config for provider: {name}")
                continue

            try:
                provider_class = PROVIDER_CLASSES[name]
                provider = provider_class(provider_config)

                if provider.is_available():
                    self._providers[name] = provider
                    self._provider_order.append(name)
                    self.logger.info(f"Initialized provider: {name}")
                else:
                    self.logger.warning(f"Provider not available: {name}")

            except Exception as e:
                self.logger.error(f"Failed to initialize {name}: {e}")

        if not self._providers:
            raise ProviderError("No providers available")

        self.logger.info(f"Active providers: {self._provider_order}")

    @property
    def current_provider(self) -> BaseProvider:
        """Get the current active provider."""
        if not self._provider_order:
            raise ProviderError("No providers available")

        name = self._provider_order[self._current_index]
        return self._providers[name]

    @property
    def current_provider_name(self) -> str:
        """Get the current provider name."""
        if not self._provider_order:
            return ""
        return self._provider_order[self._current_index]

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate content with automatic failover.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction

        Returns:
            GenerationResult from successful provider

        Raises:
            ProviderError: If all providers fail
        """
        last_error = None
        attempts = 0
        max_attempts = len(self._providers) * 2  # Allow retries

        while attempts < max_attempts:
            provider = self.current_provider
            provider_name = self.current_provider_name

            try:
                self.logger.debug(f"Attempting generation with {provider_name}")
                result = provider.generate(prompt, system_prompt)

                if result.success:
                    return result
                else:
                    self.logger.warning(f"{provider_name} returned non-success: {result.finish_reason}")
                    if self._should_failover(result.finish_reason):
                        self._switch_provider("non-success result")
                    else:
                        return result

            except RateLimitError as e:
                self.logger.warning(f"{provider_name} rate limited: {e}")
                last_error = e
                if self.config.auto_switch:
                    self._switch_provider("rate limit")
                else:
                    raise

            except AuthenticationError as e:
                self.logger.error(f"{provider_name} auth failed: {e}")
                last_error = e
                self._switch_provider("auth failure")

            except ProviderError as e:
                self.logger.warning(f"{provider_name} error: {e}")
                last_error = e
                if self.config.auto_switch:
                    self._switch_provider("provider error")
                else:
                    raise

            attempts += 1

        raise ProviderError(f"All providers failed. Last error: {last_error}")

    def _should_failover(self, finish_reason) -> bool:
        """Check if we should failover based on finish reason."""
        from .base import FinishReason
        return finish_reason in [FinishReason.ERROR, FinishReason.SAFETY]

    def _switch_provider(self, reason: str) -> bool:
        """Switch to next provider in failover chain.

        Args:
            reason: Reason for switching

        Returns:
            True if switched successfully
        """
        if len(self._provider_order) <= 1:
            self.logger.warning("No failover providers available")
            return False

        old_name = self.current_provider_name
        self._current_index = (self._current_index + 1) % len(self._provider_order)
        new_name = self.current_provider_name

        self.logger.info(f"Switched provider: {old_name} -> {new_name} ({reason})")
        return True

    def reset_to_primary(self) -> None:
        """Reset to primary provider."""
        self._current_index = 0
        self.logger.info(f"Reset to primary provider: {self.current_provider_name}")

    def get_usage(self) -> dict:
        """Get combined usage from all providers."""
        usage = {
            'total_tokens': 0,
            'total_requests': 0,
            'by_provider': {}
        }

        for name, provider in self._providers.items():
            provider_usage = provider.get_usage()
            usage['total_tokens'] += provider_usage['total_tokens']
            usage['total_requests'] += provider_usage['total_requests']
            usage['by_provider'][name] = provider_usage

        return usage

    def list_providers(self) -> List[str]:
        """List available providers in priority order."""
        return self._provider_order.copy()


def create_provider(config: Any) -> ProviderFactory:
    """Create a provider factory from configuration.

    Args:
        config: Main Config object

    Returns:
        Configured ProviderFactory
    """
    return ProviderFactory(config)


if __name__ == "__main__":
    print("=" * 60)
    print("PROVIDER FACTORY TEST")
    print("=" * 60)

    import os
    from ..core.config import load_config, _create_default_config

    # Create test config with API keys from environment
    config = _create_default_config()

    # Set API keys from environment
    if 'gemini' in config.providers:
        config.providers['gemini'].api_key = os.getenv("GEMINI_API_KEY", "")
    if 'openrouter' in config.providers:
        config.providers['openrouter'].api_key = os.getenv("OPENROUTER_API_KEY", "")

    # Test 1: Create factory
    try:
        factory = ProviderFactory(config)
        print(f"  ✓ Factory created")
        print(f"    Available providers: {factory.list_providers()}")
        print(f"    Current provider: {factory.current_provider_name}")
    except ProviderError as e:
        print(f"  ✗ Factory creation failed: {e}")
        print("  Set GEMINI_API_KEY or OPENROUTER_API_KEY to test")
        exit()

    # Test 2: Check current provider
    provider = factory.current_provider
    print(f"  ✓ Current provider: {provider.name}")

    # Test 3: Provider switching
    if len(factory.list_providers()) > 1:
        old_name = factory.current_provider_name
        factory._switch_provider("test")
        new_name = factory.current_provider_name
        print(f"  ✓ Provider switch: {old_name} -> {new_name}")

        # Reset
        factory.reset_to_primary()
        print(f"  ✓ Reset to primary: {factory.current_provider_name}")
    else:
        print("  ⚠ Only one provider available, skipping switch test")

    # Test 4: Generation (if API key available)
    if factory.current_provider.api_key:
        try:
            result = factory.generate("Say 'test' and nothing else.")
            print(f"  ✓ Generation successful: '{result.text[:30]}...'")
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")
    else:
        print("  ⚠ No API key - skipping generation test")

    # Test 5: Usage stats
    usage = factory.get_usage()
    print(f"  ✓ Usage stats: {usage['total_requests']} requests")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
