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
    if name == 'OpenAIProvider':
        from .openai_provider import OpenAIProvider
        return OpenAIProvider
    if name == 'ProviderFactory':
        from .factory import ProviderFactory
        return ProviderFactory
    if name == 'LoadBalancer':
        from .load_balancer import LoadBalancer
        return LoadBalancer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['BaseProvider', 'GeminiProvider', 'OpenRouterProvider', 'OpenAIProvider', 'ProviderFactory', 'LoadBalancer']
