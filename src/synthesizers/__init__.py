"""Synthesis engines for different content types."""

def __getattr__(name):
    if name == 'BaseSynthesizer':
        from .base import BaseSynthesizer
        return BaseSynthesizer
    if name == 'QASynthesizer':
        from .qa_synthesis import QASynthesizer
        return QASynthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['BaseSynthesizer', 'QASynthesizer']
