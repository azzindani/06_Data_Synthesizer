#!/usr/bin/env python3
"""Performance benchmarks for the Data Synthesizer."""

import os
import sys
import time
import json
import argparse
import statistics
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import load_config, _create_default_config
from src.core.logger import get_logger, setup_root_logger
from src.providers.factory import ProviderFactory, create_provider
from src.utils.json_parser import extract_json, parse_json_safely


def benchmark_provider_latency(provider, num_requests: int = 10) -> Dict[str, Any]:
    """Benchmark provider response latency.

    Args:
        provider: Provider instance
        num_requests: Number of test requests

    Returns:
        Latency statistics
    """
    latencies = []
    errors = 0
    test_prompt = "Reply with only the word 'test'."

    for i in range(num_requests):
        start = time.time()
        try:
            result = provider.generate(test_prompt)
            if result.success:
                latencies.append(time.time() - start)
            else:
                errors += 1
        except Exception as e:
            errors += 1

        # Small delay between requests
        time.sleep(0.5)

    if latencies:
        return {
            'requests': num_requests,
            'successful': len(latencies),
            'errors': errors,
            'mean_ms': statistics.mean(latencies) * 1000,
            'median_ms': statistics.median(latencies) * 1000,
            'min_ms': min(latencies) * 1000,
            'max_ms': max(latencies) * 1000,
            'stdev_ms': statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0
        }
    else:
        return {
            'requests': num_requests,
            'successful': 0,
            'errors': errors,
            'mean_ms': 0,
            'median_ms': 0,
            'min_ms': 0,
            'max_ms': 0,
            'stdev_ms': 0
        }


def benchmark_throughput(provider, duration_seconds: int = 60) -> Dict[str, Any]:
    """Benchmark provider throughput.

    Args:
        provider: Provider instance
        duration_seconds: Test duration

    Returns:
        Throughput statistics
    """
    test_prompt = "Generate a short response about data synthesis."

    start_time = time.time()
    requests = 0
    tokens = 0
    errors = 0

    while time.time() - start_time < duration_seconds:
        try:
            result = provider.generate(test_prompt)
            if result.success:
                requests += 1
                tokens += result.input_tokens + result.output_tokens
            else:
                errors += 1
        except Exception:
            errors += 1

        # Respect rate limits
        time.sleep(0.5)

    elapsed = time.time() - start_time

    return {
        'duration_seconds': elapsed,
        'total_requests': requests,
        'total_tokens': tokens,
        'errors': errors,
        'requests_per_minute': (requests / elapsed) * 60,
        'tokens_per_minute': (tokens / elapsed) * 60
    }


def benchmark_json_parsing(num_iterations: int = 1000) -> Dict[str, Any]:
    """Benchmark JSON parsing performance.

    Args:
        num_iterations: Number of parsing iterations

    Returns:
        Parsing statistics
    """
    # Test cases
    test_cases = [
        # Clean JSON
        '[{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]',
        # JSON in markdown
        '```json\n[{"question": "Q1", "answer": "A1"}]\n```',
        # Malformed JSON
        '{"question": "Q1", "answer": "A1",}',
        # JSON with extra text
        'Here is the result:\n[{"question": "Q1", "answer": "A1"}]\nEnd.',
    ]

    times = []

    for _ in range(num_iterations):
        for case in test_cases:
            start = time.time()
            try:
                result = extract_json(case)
                if result:
                    parse_json_safely(result)
            except Exception:
                pass
            times.append(time.time() - start)

    return {
        'iterations': num_iterations * len(test_cases),
        'mean_us': statistics.mean(times) * 1_000_000,
        'median_us': statistics.median(times) * 1_000_000,
        'max_us': max(times) * 1_000_000,
        'total_ms': sum(times) * 1000
    }


def benchmark_memory_usage() -> Dict[str, Any]:
    """Benchmark memory usage.

    Returns:
        Memory statistics
    """
    import tracemalloc

    tracemalloc.start()

    # Simulate typical operations
    from src.core.config import _create_default_config
    from src.core.progress import ProgressManager

    config = _create_default_config()

    # Create progress manager with some data
    pm = ProgressManager(config)
    for i in range(100):
        pm.mark_processed(f"item_{i}", {'generated': 5})

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'current_mb': current / 1024 / 1024,
        'peak_mb': peak / 1024 / 1024
    }


def run_benchmarks(config: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Run all benchmarks.

    Args:
        config: Configuration object
        args: Command line arguments

    Returns:
        Benchmark results
    """
    logger = get_logger(__name__)
    results = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {}
    }

    # Provider latency benchmark
    if not args.skip_api:
        logger.info("Running provider latency benchmark...")
        try:
            factory = create_provider(config)
            results['benchmarks']['latency'] = benchmark_provider_latency(
                factory.current_provider,
                num_requests=args.num_requests
            )
            logger.info(f"  Mean latency: {results['benchmarks']['latency']['mean_ms']:.0f}ms")
        except Exception as e:
            logger.error(f"Latency benchmark failed: {e}")
            results['benchmarks']['latency'] = {'error': str(e)}

        # Throughput benchmark
        if args.throughput:
            logger.info("Running throughput benchmark...")
            try:
                results['benchmarks']['throughput'] = benchmark_throughput(
                    factory.current_provider,
                    duration_seconds=args.duration
                )
                logger.info(f"  Requests/min: {results['benchmarks']['throughput']['requests_per_minute']:.1f}")
            except Exception as e:
                logger.error(f"Throughput benchmark failed: {e}")
                results['benchmarks']['throughput'] = {'error': str(e)}

    # JSON parsing benchmark
    logger.info("Running JSON parsing benchmark...")
    results['benchmarks']['json_parsing'] = benchmark_json_parsing(
        num_iterations=args.parsing_iterations
    )
    logger.info(f"  Mean parse time: {results['benchmarks']['json_parsing']['mean_us']:.1f}µs")

    # Memory benchmark
    logger.info("Running memory benchmark...")
    results['benchmarks']['memory'] = benchmark_memory_usage()
    logger.info(f"  Peak memory: {results['benchmarks']['memory']['peak_mb']:.2f}MB")

    return results


def print_results(results: Dict[str, Any]):
    """Print benchmark results in a readable format."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Timestamp: {results['timestamp']}\n")

    for name, data in results['benchmarks'].items():
        print(f"📊 {name.upper()}")
        if 'error' in data:
            print(f"   Error: {data['error']}")
        else:
            for key, value in data.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        print()

    print("=" * 60)


def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Data Synthesizer Benchmarks")
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--output', default='benchmark_results.json', help='Output file')
    parser.add_argument('--num-requests', type=int, default=10, help='Latency test requests')
    parser.add_argument('--duration', type=int, default=60, help='Throughput test duration')
    parser.add_argument('--parsing-iterations', type=int, default=1000, help='JSON parsing iterations')
    parser.add_argument('--throughput', action='store_true', help='Run throughput test')
    parser.add_argument('--skip-api', action='store_true', help='Skip API-based tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    setup_root_logger('INFO' if not args.verbose else 'DEBUG')
    logger = get_logger(__name__)

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        logger.warning(f"Config not found: {args.config}, using defaults")
        config = _create_default_config()

    # Run benchmarks
    logger.info("Starting benchmarks...")
    results = run_benchmarks(config, args)

    # Print results
    print_results(results)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
