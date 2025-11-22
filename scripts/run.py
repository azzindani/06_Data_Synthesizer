#!/usr/bin/env python3
"""Main entry point for data synthesizer."""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import load_config
from src.core.logger import setup_root_logger
from src.synthesizers.qa_synthesis import QASynthesizer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Data Synthesizer - Generate synthetic datasets using LLMs"
    )

    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--type', '-t',
        choices=['qa', 'deep_thinking', 'corpus'],
        help='Override synthesis type from config'
    )

    parser.add_argument(
        '--provider', '-p',
        choices=['gemini', 'openrouter', 'openai'],
        help='Override primary provider'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making API calls (for testing)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Apply overrides
    if args.type:
        config.synthesis.type = args.type

    if args.provider:
        config.primary_provider = args.provider

    if args.verbose:
        config.log_level = 'DEBUG'

    # Setup logging
    logger = setup_root_logger(config.log_file, config.log_level)
    logger.info("Data Synthesizer starting...")

    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        sys.exit(1)

    # Log configuration
    logger.info(f"Synthesis type: {config.synthesis.type}")
    logger.info(f"Primary provider: {config.primary_provider}")
    logger.info(f"Output: {config.output.type} -> {config.output.repository or config.output.local_path}")

    if args.dry_run:
        logger.info("DRY RUN MODE - No API calls will be made")
        print("\nConfiguration validated successfully!")
        print(f"  Type: {config.synthesis.type}")
        print(f"  Provider: {config.primary_provider}")
        print(f"  Topics: {len(config.domain.get('topics', []))}")
        print(f"  Variants per topic: {config.synthesis.num_variants}")
        print(f"  Questions per topic: {config.synthesis.questions_per_topic}")
        return

    # Create synthesizer based on type
    synthesis_type = config.synthesis.type

    if synthesis_type == 'qa':
        synthesizer = QASynthesizer(config)
    elif synthesis_type == 'deep_thinking':
        # TODO: Implement DeepThinkingSynthesizer
        logger.error("Deep thinking synthesizer not yet implemented")
        sys.exit(1)
    elif synthesis_type == 'corpus':
        # TODO: Implement CorpusGenerator
        logger.error("Corpus generator not yet implemented")
        sys.exit(1)
    else:
        logger.error(f"Unknown synthesis type: {synthesis_type}")
        sys.exit(1)

    # Show initial progress
    print("\nInitial Progress:")
    synthesizer.show_progress()

    # Run synthesis
    print("\nStarting synthesis...")
    try:
        synthesizer.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        sys.exit(1)

    # Final progress
    print("\nFinal Progress:")
    synthesizer.show_progress()

    logger.info("Data Synthesizer completed")


if __name__ == '__main__':
    main()
