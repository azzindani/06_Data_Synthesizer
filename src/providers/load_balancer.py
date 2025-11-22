"""Dynamic provider load balancing based on performance metrics."""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core.logger import get_logger
from .base import BaseProvider, GenerationResult, ProviderError


@dataclass
class ProviderMetrics:
    """Performance metrics for a provider."""
    name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    rate_limit_hits: int = 0
    last_error_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return float('inf')
        return self.total_latency_ms / self.successful_requests

    @property
    def score(self) -> float:
        """Calculate overall provider score (higher is better)."""
        # Weighted score based on success rate and latency
        success_weight = 0.7
        latency_weight = 0.3

        success_score = self.success_rate * 100

        # Normalize latency (lower is better, cap at 10s)
        latency_score = max(0, 100 - (self.avg_latency_ms / 100))

        return (success_weight * success_score) + (latency_weight * latency_score)

    def is_available(self) -> bool:
        """Check if provider is available (not in cooldown)."""
        if self.cooldown_until is None:
            return True
        return datetime.now() >= self.cooldown_until


class LoadBalancer:
    """Dynamic load balancer for multiple providers."""

    def __init__(self, providers: Dict[str, BaseProvider], strategy: str = "adaptive"):
        """Initialize load balancer.

        Args:
            providers: Dictionary of provider name to provider instance
            strategy: Load balancing strategy (adaptive, round_robin, weighted)
        """
        self.logger = get_logger(__name__)
        self._providers = providers
        self._strategy = strategy

        # Initialize metrics for each provider
        self._metrics: Dict[str, ProviderMetrics] = {
            name: ProviderMetrics(name=name)
            for name in providers.keys()
        }

        # Round robin index
        self._rr_index = 0
        self._provider_order = list(providers.keys())

        self.logger.info(f"Load balancer initialized with strategy: {strategy}")
        self.logger.info(f"Providers: {list(providers.keys())}")

    def get_provider(self) -> tuple[str, BaseProvider]:
        """Get the best available provider based on strategy.

        Returns:
            Tuple of (provider_name, provider_instance)

        Raises:
            ProviderError: If no providers available
        """
        if self._strategy == "round_robin":
            return self._round_robin()
        elif self._strategy == "weighted":
            return self._weighted_selection()
        else:  # adaptive
            return self._adaptive_selection()

    def _round_robin(self) -> tuple[str, BaseProvider]:
        """Simple round-robin selection."""
        available = [
            name for name in self._provider_order
            if self._metrics[name].is_available()
        ]

        if not available:
            raise ProviderError("No providers available")

        # Find next available provider
        for _ in range(len(self._provider_order)):
            name = self._provider_order[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(self._provider_order)
            if name in available:
                return name, self._providers[name]

        raise ProviderError("No providers available")

    def _weighted_selection(self) -> tuple[str, BaseProvider]:
        """Select provider based on weighted scores."""
        available = [
            (name, self._metrics[name])
            for name in self._provider_order
            if self._metrics[name].is_available()
        ]

        if not available:
            raise ProviderError("No providers available")

        # Sort by score (highest first)
        available.sort(key=lambda x: x[1].score, reverse=True)

        name = available[0][0]
        return name, self._providers[name]

    def _adaptive_selection(self) -> tuple[str, BaseProvider]:
        """Adaptive selection based on real-time performance."""
        available = []

        for name in self._provider_order:
            metrics = self._metrics[name]
            if not metrics.is_available():
                continue

            # Calculate adaptive score
            score = metrics.score

            # Boost score for underutilized providers (exploration)
            if metrics.total_requests < 10:
                score += 20

            # Penalize recent errors
            if metrics.last_error_time:
                time_since_error = (datetime.now() - metrics.last_error_time).total_seconds()
                if time_since_error < 60:
                    score -= 10

            available.append((name, score))

        if not available:
            raise ProviderError("No providers available")

        # Sort by score and select best
        available.sort(key=lambda x: x[1], reverse=True)

        name = available[0][0]
        self.logger.debug(f"Selected provider: {name} (score: {available[0][1]:.1f})")
        return name, self._providers[name]

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate content using best available provider.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction

        Returns:
            GenerationResult from provider
        """
        last_error = None

        for attempt in range(len(self._providers)):
            try:
                name, provider = self.get_provider()
                metrics = self._metrics[name]

                # Record request start
                start_time = time.time()

                try:
                    result = provider.generate(prompt, system_prompt)
                    latency_ms = (time.time() - start_time) * 1000

                    # Update metrics
                    metrics.total_requests += 1
                    if result.success:
                        metrics.successful_requests += 1
                        metrics.total_latency_ms += latency_ms
                    else:
                        metrics.failed_requests += 1

                    return result

                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000

                    # Update failure metrics
                    metrics.total_requests += 1
                    metrics.failed_requests += 1
                    metrics.last_error_time = datetime.now()

                    # Check if rate limited
                    if "rate" in str(e).lower() or "429" in str(e):
                        metrics.rate_limit_hits += 1
                        # Set cooldown
                        cooldown_seconds = min(60 * (2 ** metrics.rate_limit_hits), 300)
                        metrics.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
                        self.logger.warning(f"{name} rate limited, cooldown: {cooldown_seconds}s")

                    last_error = e
                    self.logger.warning(f"{name} failed: {e}")

            except ProviderError:
                # No providers available
                break

        raise ProviderError(f"All providers failed. Last error: {last_error}")

    def record_result(self, provider_name: str, success: bool, latency_ms: float):
        """Manually record a result for a provider.

        Args:
            provider_name: Name of the provider
            success: Whether the request succeeded
            latency_ms: Request latency in milliseconds
        """
        if provider_name not in self._metrics:
            return

        metrics = self._metrics[provider_name]
        metrics.total_requests += 1

        if success:
            metrics.successful_requests += 1
            metrics.total_latency_ms += latency_ms
        else:
            metrics.failed_requests += 1
            metrics.last_error_time = datetime.now()

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all providers.

        Returns:
            Dictionary of provider metrics
        """
        return {
            name: {
                'total_requests': m.total_requests,
                'successful_requests': m.successful_requests,
                'failed_requests': m.failed_requests,
                'success_rate': m.success_rate,
                'avg_latency_ms': m.avg_latency_ms if m.successful_requests > 0 else 0,
                'score': m.score,
                'available': m.is_available(),
                'rate_limit_hits': m.rate_limit_hits
            }
            for name, m in self._metrics.items()
        }

    def reset_metrics(self):
        """Reset all provider metrics."""
        for name in self._metrics:
            self._metrics[name] = ProviderMetrics(name=name)
        self.logger.info("Metrics reset")

    def set_strategy(self, strategy: str):
        """Change the load balancing strategy.

        Args:
            strategy: New strategy (adaptive, round_robin, weighted)
        """
        if strategy not in ['adaptive', 'round_robin', 'weighted']:
            raise ValueError(f"Unknown strategy: {strategy}")
        self._strategy = strategy
        self.logger.info(f"Strategy changed to: {strategy}")


if __name__ == "__main__":
    print("=" * 60)
    print("LOAD BALANCER TEST")
    print("=" * 60)

    from unittest.mock import Mock

    # Create mock providers
    providers = {}
    for name in ['gemini', 'openrouter', 'openai']:
        mock = Mock(spec=BaseProvider)
        mock.name = name
        providers[name] = mock

    # Test 1: Create balancer
    balancer = LoadBalancer(providers, strategy="adaptive")
    print(f"  ✓ Created load balancer")

    # Test 2: Get provider
    name, provider = balancer.get_provider()
    print(f"  ✓ Got provider: {name}")

    # Test 3: Record metrics
    balancer.record_result("gemini", True, 500)
    balancer.record_result("gemini", True, 600)
    balancer.record_result("openrouter", True, 300)
    print(f"  ✓ Recorded metrics")

    # Test 4: Get metrics
    metrics = balancer.get_metrics()
    print(f"  ✓ Gemini score: {metrics['gemini']['score']:.1f}")
    print(f"  ✓ OpenRouter score: {metrics['openrouter']['score']:.1f}")

    # Test 5: Change strategy
    balancer.set_strategy("round_robin")
    name1, _ = balancer.get_provider()
    name2, _ = balancer.get_provider()
    print(f"  ✓ Round robin: {name1} -> {name2}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
