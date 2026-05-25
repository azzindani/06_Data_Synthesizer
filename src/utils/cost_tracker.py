"""Cost tracking for LLM API usage."""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime

from ..core.logger import get_logger

# Per-request records are only used by ``get_recent_records(N)``. Capping
# the buffer keeps memory flat over a 100k+ request cron run — older
# records are dropped on overflow; the running totals dict keeps the
# aggregates that actually matter.
_RECORDS_BUFFER_SIZE = 1000


@dataclass
class ProviderPricing:
    """Pricing information for a provider."""
    name: str
    input_cost_per_1k: float  # USD per 1000 tokens
    output_cost_per_1k: float  # USD per 1000 tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost


# Default pricing (as of 2024, update as needed)
DEFAULT_PRICING = {
    'gemini': ProviderPricing(
        name='gemini',
        input_cost_per_1k=0.000125,  # Gemini Flash
        output_cost_per_1k=0.000375
    ),
    'openrouter': ProviderPricing(
        name='openrouter',
        input_cost_per_1k=0.00015,  # Varies by model
        output_cost_per_1k=0.0006
    ),
    'openai': ProviderPricing(
        name='openai',
        input_cost_per_1k=0.00015,  # GPT-4o-mini
        output_cost_per_1k=0.0006
    )
}


@dataclass
class UsageRecord:
    """Record of a single API usage."""
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    success: bool


class CostTracker:
    """Tracks API usage costs across providers."""

    def __init__(self, pricing: Dict[str, ProviderPricing] = None):
        """Initialize cost tracker.

        Args:
            pricing: Custom pricing dict, uses defaults if not provided
        """
        self.logger = get_logger(__name__)
        self.pricing = pricing or DEFAULT_PRICING
        # Bounded ring buffer — preserves the last N records for
        # get_recent_records() while keeping memory flat regardless of run
        # length. The totals dict below remains the source of truth for
        # aggregate stats.
        self.records: "deque[UsageRecord]" = deque(maxlen=_RECORDS_BUFFER_SIZE)

        # Running totals
        self.totals = {
            'requests': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'by_provider': {}
        }

    def record_usage(self, provider: str, model: str,
                     input_tokens: int, output_tokens: int,
                     success: bool = True) -> float:
        """Record API usage and return cost.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            success: Whether the call succeeded

        Returns:
            Cost in USD
        """
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        if provider in self.pricing:
            cost = self.pricing[provider].calculate_cost(input_tokens, output_tokens)
        else:
            # Default estimate
            cost = (total_tokens / 1000) * 0.001
            self.logger.warning(f"No pricing for {provider}, using estimate")

        # Create record
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            success=success
        )
        self.records.append(record)

        # Update totals
        self.totals['requests'] += 1
        self.totals['input_tokens'] += input_tokens
        self.totals['output_tokens'] += output_tokens
        self.totals['total_tokens'] += total_tokens
        self.totals['total_cost'] += cost

        # Update by-provider totals
        if provider not in self.totals['by_provider']:
            self.totals['by_provider'][provider] = {
                'requests': 0,
                'tokens': 0,
                'cost': 0.0
            }
        self.totals['by_provider'][provider]['requests'] += 1
        self.totals['by_provider'][provider]['tokens'] += total_tokens
        self.totals['by_provider'][provider]['cost'] += cost

        return cost

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary.

        Returns:
            Summary dictionary
        """
        return {
            'total_requests': self.totals['requests'],
            'total_tokens': self.totals['total_tokens'],
            'total_cost_usd': round(self.totals['total_cost'], 4),
            'avg_cost_per_request': round(
                self.totals['total_cost'] / self.totals['requests'], 6
            ) if self.totals['requests'] > 0 else 0,
            'by_provider': {
                name: {
                    'requests': data['requests'],
                    'tokens': data['tokens'],
                    'cost_usd': round(data['cost'], 4)
                }
                for name, data in self.totals['by_provider'].items()
            }
        }

    def get_recent_records(self, limit: int = 10) -> List[Dict]:
        """Get recent usage records.

        Args:
            limit: Number of records to return

        Returns:
            List of record dictionaries
        """
        # deque doesn't support slicing — materialise just the tail.
        if len(self.records) > limit:
            recent = list(self.records)[-limit:]
        else:
            recent = list(self.records)
        return [
            {
                'timestamp': r.timestamp,
                'provider': r.provider,
                'tokens': r.total_tokens,
                'cost': round(r.cost, 6)
            }
            for r in recent
        ]

    def estimate_remaining_cost(self, remaining_items: int,
                                avg_tokens_per_item: int = 2000) -> float:
        """Estimate cost for remaining items.

        Args:
            remaining_items: Number of items left to process
            avg_tokens_per_item: Estimated tokens per item

        Returns:
            Estimated cost in USD
        """
        if self.totals['requests'] == 0:
            # Use default estimate
            return (remaining_items * avg_tokens_per_item / 1000) * 0.001

        # Calculate average cost per request
        avg_cost = self.totals['total_cost'] / self.totals['requests']
        return remaining_items * avg_cost

    def format_cost(self, cost: float) -> str:
        """Format cost for display.

        Args:
            cost: Cost in USD

        Returns:
            Formatted string
        """
        if cost < 0.01:
            return f"${cost:.4f}"
        elif cost < 1:
            return f"${cost:.3f}"
        else:
            return f"${cost:.2f}"

    def print_summary(self) -> None:
        """Print formatted cost summary."""
        summary = self.get_summary()

        print("\n" + "=" * 50)
        print("COST SUMMARY")
        print("=" * 50)
        print(f"Total Requests: {summary['total_requests']:,}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"Total Cost: {self.format_cost(summary['total_cost_usd'])}")

        if summary['by_provider']:
            print("\nBy Provider:")
            for name, data in summary['by_provider'].items():
                print(f"  {name}:")
                print(f"    Requests: {data['requests']:,}")
                print(f"    Tokens: {data['tokens']:,}")
                print(f"    Cost: {self.format_cost(data['cost_usd'])}")

        print("=" * 50)


if __name__ == "__main__":
    print("=" * 60)
    print("COST TRACKER TEST")
    print("=" * 60)

    tracker = CostTracker()

    # Test 1: Record usage
    cost1 = tracker.record_usage('gemini', 'gemini-flash', 1000, 500)
    print(f"  ✓ Gemini call cost: {tracker.format_cost(cost1)}")

    cost2 = tracker.record_usage('openai', 'gpt-4o-mini', 800, 400)
    print(f"  ✓ OpenAI call cost: {tracker.format_cost(cost2)}")

    cost3 = tracker.record_usage('gemini', 'gemini-flash', 1200, 600)
    print(f"  ✓ Gemini call cost: {tracker.format_cost(cost3)}")

    # Test 2: Get summary
    summary = tracker.get_summary()
    print(f"  ✓ Total cost: {tracker.format_cost(summary['total_cost_usd'])}")
    print(f"  ✓ Total requests: {summary['total_requests']}")

    # Test 3: Estimate remaining
    estimate = tracker.estimate_remaining_cost(100)
    print(f"  ✓ Estimated cost for 100 more items: {tracker.format_cost(estimate)}")

    # Test 4: Print full summary
    tracker.print_summary()

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
