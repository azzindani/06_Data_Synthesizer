#!/usr/bin/env python3
"""Test provider integrations - run with: python -m scripts.test_providers"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import _create_default_config
from src.core.logger import setup_root_logger
from src.providers.base import BaseProvider, FinishReason, GenerationResult
from src.providers.gemini import GeminiProvider
from src.providers.openrouter import OpenRouterProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.load_balancer import LoadBalancer, ProviderMetrics


def test_provider_metrics():
    """Test provider metrics calculation."""
    print("\n" + "=" * 50)
    print("TEST: Provider Metrics")
    print("=" * 50)

    metrics = ProviderMetrics(name='test')
    metrics.total_requests = 100
    metrics.successful_requests = 95
    metrics.failed_requests = 5
    metrics.total_latency_ms = 50000

    assert metrics.success_rate == 0.95
    # avg_latency = total_latency / successful_requests = 50000 / 95
    assert abs(metrics.avg_latency_ms - 526.32) < 1
    assert metrics.score > 0

    print(f"  ✓ Success rate: {metrics.success_rate * 100}%")
    print(f"  ✓ Avg latency: {metrics.avg_latency_ms:.0f}ms")
    print(f"  ✓ Score: {metrics.score:.1f}")

    return True


def test_load_balancer():
    """Test load balancer strategies."""
    print("\n" + "=" * 50)
    print("TEST: Load Balancer")
    print("=" * 50)

    from unittest.mock import Mock

    # Create mock providers
    providers = {
        'gemini': Mock(spec=BaseProvider),
        'openai': Mock(spec=BaseProvider)
    }

    # Test adaptive strategy
    balancer = LoadBalancer(providers, strategy='adaptive')
    name, provider = balancer.get_provider()
    assert name in providers
    print(f"  ✓ Adaptive: selected {name}")

    # Test round robin
    balancer.set_strategy('round_robin')
    names = []
    for _ in range(4):
        n, _ = balancer.get_provider()
        names.append(n)
    print(f"  ✓ Round robin: {names}")

    # Test metrics recording
    balancer.record_result('gemini', True, 500)
    balancer.record_result('gemini', True, 600)
    balancer.record_result('openai', False, 0)

    metrics = balancer.get_metrics()
    print(f"  ✓ Gemini requests: {metrics['gemini']['total_requests']}")
    print(f"  ✓ OpenAI failures: {metrics['openai']['failed_requests']}")

    return True


def test_provider_availability():
    """Test provider availability checks."""
    print("\n" + "=" * 50)
    print("TEST: Provider Availability")
    print("=" * 50)

    config = _create_default_config()

    # Test Gemini
    gemini_config = config.providers.get('gemini')
    if gemini_config:
        gemini = GeminiProvider(gemini_config)
        available = gemini.is_available()
        print(f"  - Gemini: {'✓ Available' if available else '✗ Not available (no API key)'}")

    # Test OpenRouter
    openrouter_config = config.providers.get('openrouter')
    if openrouter_config:
        openrouter = OpenRouterProvider(openrouter_config)
        available = openrouter.is_available()
        print(f"  - OpenRouter: {'✓ Available' if available else '✗ Not available (no API key)'}")

    # Test OpenAI
    openai_config = config.providers.get('openai')
    if openai_config:
        openai = OpenAIProvider(openai_config)
        available = openai.is_available()
        print(f"  - OpenAI: {'✓ Available' if available else '✗ Not available (no API key)'}")

    return True


def test_finish_reasons():
    """Test finish reason mappings."""
    print("\n" + "=" * 50)
    print("TEST: Finish Reason Mappings")
    print("=" * 50)

    # Test all finish reasons
    reasons = {
        FinishReason.COMPLETE: True,
        FinishReason.MAX_TOKENS: True,
        FinishReason.SAFETY: False,
        FinishReason.RECITATION: False,
        FinishReason.ERROR: False,
    }

    for reason, expected_success in reasons.items():
        result = GenerationResult(text="test", finish_reason=reason)
        actual = result.success
        status = "✓" if actual == expected_success else "✗"
        print(f"  {status} {reason.name}: success={actual}")

    return True


def main():
    """Run all provider tests."""
    setup_root_logger('WARNING')

    print("=" * 60)
    print("DATA SYNTHESIZER - PROVIDER TESTS")
    print("=" * 60)

    tests = [
        ("Provider Metrics", test_provider_metrics),
        ("Load Balancer", test_load_balancer),
        ("Provider Availability", test_provider_availability),
        ("Finish Reasons", test_finish_reasons),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
