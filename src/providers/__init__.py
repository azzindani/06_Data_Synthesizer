"""LLM provider implementations."""

def __getattr__(name):
    if name == 'BaseProvider':
        from .base import BaseProvider
        return BaseProvider
    if name == 'GeminiProvider':
        from .gemini import GeminiProvider
        return GeminiProvider
    if name == 'OpenRouterProvider':
        from .openrouter import OpenRouterProvider
        return OpenRouterProvider
    if name == 'ProviderFactory':
        from .factory import ProviderFactory
        return ProviderFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['BaseProvider', 'GeminiProvider', 'OpenRouterProvider', 'ProviderFactory']
