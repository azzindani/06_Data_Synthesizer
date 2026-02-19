"""Base synthesizer class with common functionality."""

import signal
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Dict, Optional

from ..core.config import Config
from ..core.logger import get_logger
from ..core.progress import ProgressManager
from ..providers.factory import ProviderFactory
from ..providers.base import ProviderError, SafetyFilterError
from ..utils.json_parser import parse_qa_variants


class BaseSynthesizer(ABC):
    """Abstract base class for all synthesizers."""

    def __init__(self, config: Config, provider: Optional[ProviderFactory] = None):
        """Initialize synthesizer.

        Args:
            config: Configuration object
            provider: Optional provider factory (created if not provided)
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Initialize provider
        if provider:
            self.provider = provider
        else:
            self.provider = ProviderFactory(config)

        # Initialize progress manager
        self.progress = ProgressManager(config)

        # Shutdown flag
        self._shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def handle_shutdown(signum, frame):
            self.logger.info("Shutdown signal received, finishing current item...")
            self._shutdown_requested = True

        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

    @abstractmethod
    def create_prompt(self, item: Dict) -> str:
        """Create synthesis prompt for an item.

        Args:
            item: Item data to synthesize from

        Returns:
            Prompt string
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get system prompt for this synthesizer.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_topics(self) -> List[str]:
        """Get list of topics to synthesize.

        Returns:
            List of topic strings
        """
        pass

    def synthesize_item(self, item: Dict, item_id: str) -> List[Dict]:
        """Synthesize a single item.

        Args:
            item: Item data
            item_id: Unique identifier

        Returns:
            List of generated results
        """
        # Check if already processed
        if self.progress.is_processed(item_id):
            self.logger.debug(f"Item {item_id} already processed, skipping")
            return []

        # Create prompt
        prompt = self.create_prompt(item)
        system_prompt = self.get_system_prompt()

        try:
            # Generate content
            result = self.provider.generate(prompt, system_prompt)
            self.progress.increment_requests()

            if not result.success:
                self.logger.warning(f"Generation failed: {result.finish_reason}")
                self.progress.add_error({
                    'type': 'generation_failed',
                    'item_id': item_id,
                    'reason': str(result.finish_reason)
                })
                return []

            # Parse response
            variants = parse_qa_variants(result.text)

            if not variants:
                self.logger.warning(f"No valid variants parsed for {item_id}")
                self.progress.add_error({
                    'type': 'parse_failed',
                    'item_id': item_id,
                    'response_preview': result.text[:200]
                })
                return []

            # Process variants
            results = self._process_variants(variants, item, item_id)

            # Update progress
            if results:
                stats = self._calculate_stats(results)
                self.progress.mark_processed(item_id, stats)
                self.progress.add_generated(len(results))

            return results

        except SafetyFilterError as e:
            self.logger.warning(f"Safety filter for {item_id}: {e}")
            self.progress.add_error({
                'type': 'safety_filter',
                'item_id': item_id,
                'message': str(e)
            })
            return []

        except ProviderError as e:
            self.logger.error(f"Provider error for {item_id}: {e}")
            self.progress.add_error({
                'type': 'provider_error',
                'item_id': item_id,
                'message': str(e)
            })
            return []

    def _process_variants(self, variants: List[Dict], original: Dict, item_id: str) -> List[Dict]:
        """Process and validate generated variants.

        Column names for question/answer/thinking come from
        ``config.run_config`` so they can be overridden per domain config.

        Args:
            variants: List of generated variants
            original: Original item data
            item_id: Item identifier

        Returns:
            List of processed results
        """
        results = []
        quality_config = self.config.quality
        rc = self.config.run_config

        q_col = rc.question_column   # e.g. "question" or "prompt"
        a_col = rc.answer_column     # e.g. "answer" or "response"
        t_col = rc.thinking_column   # e.g. "thinking" or "reasoning"

        for i, variant in enumerate(variants):
            question = variant.get('question', '').strip()
            answer = variant.get('answer', '').strip()
            thinking = variant.get('thinking', '').strip()

            # Basic validation
            if len(question) < quality_config.min_question_length:
                continue
            if len(answer) < quality_config.min_answer_length:
                continue

            # Truncate if too long
            if len(question) > quality_config.max_question_length:
                question = question[:quality_config.max_question_length]
            if len(answer) > quality_config.max_answer_length:
                answer = answer[:quality_config.max_answer_length]

            result = {
                'item_id': item_id,
                'variant_number': i + 1,
                q_col: question,
                a_col: answer,
                t_col: thinking,
                f'{q_col}_length': len(question),
                f'{a_col}_length': len(answer),
                'topic': original.get('topic', ''),
                'timestamp': datetime.now().isoformat(),
                'provider': self.provider.current_provider_name
            }

            results.append(result)

        return results

    def _calculate_stats(self, results: List[Dict]) -> Dict:
        """Calculate statistics for a batch of results.

        Args:
            results: List of generated results

        Returns:
            Statistics dictionary
        """
        if not results:
            return {}

        rc = self.config.run_config
        q_len_key = f'{rc.question_column}_length'
        a_len_key = f'{rc.answer_column}_length'

        stats = {
            'count': len(results),
            'avg_question_length': sum(r.get(q_len_key, 0) for r in results) / len(results),
            'avg_answer_length': sum(r.get(a_len_key, 0) for r in results) / len(results),
            'provider': results[0].get('provider', ''),
            'requests': 1
        }

        return stats

    def run(self, save_interval: int = 5) -> None:
        """Run continuous synthesis loop.

        Args:
            save_interval: Save progress every N items
        """
        topics = self.get_topics()
        total_topics = len(topics)
        items_per_topic = self.config.synthesis.questions_per_topic

        self.logger.info(f"Starting synthesis: {total_topics} topics, {items_per_topic} items each")
        self.progress.show_progress(total_topics * items_per_topic)

        for topic_idx, topic in enumerate(topics):
            if self._shutdown_requested:
                self.logger.info("Shutdown requested, stopping")
                break

            self.logger.info(f"Processing topic {topic_idx + 1}/{total_topics}: {topic}")

            # Get current progress for this topic
            topic_progress = self.progress.progress_data.get('processed_topics', {}).get(topic, {})
            completed = topic_progress.get('completed', 0)

            if completed >= items_per_topic:
                self.logger.info(f"Topic {topic} already completed")
                continue

            # Generate items for this topic
            items_generated = 0
            while items_generated < items_per_topic - completed:
                if self._shutdown_requested:
                    break

                item = {'topic': topic}
                item_id = f"{topic}_{completed + items_generated}"

                results = self.synthesize_item(item, item_id)

                if results:
                    items_generated += len(results)

                    # Save results
                    self._save_results(results, topic)

                    # Update topic progress
                    self.progress.mark_topic_progress(
                        topic,
                        completed + items_generated,
                        items_per_topic
                    )

                # Periodic progress save
                if items_generated % save_interval == 0:
                    self.progress.save()

            self.progress.save()

        self.logger.info("Synthesis complete")
        self.progress.show_progress(total_topics * items_per_topic)

    def _save_results(self, results: List[Dict], topic: str) -> None:
        """Save results (to be overridden by subclasses).

        Args:
            results: Results to save
            topic: Topic name
        """
        # Default implementation - subclasses should override
        pass

    def show_progress(self) -> None:
        """Display current progress."""
        topics = self.get_topics()
        total = len(topics) * self.config.synthesis.questions_per_topic
        self.progress.show_progress(total)


if __name__ == "__main__":
    print("=" * 60)
    print("BASE SYNTHESIZER TEST")
    print("=" * 60)

    # Note: BaseSynthesizer is abstract, so we just test that it can't be instantiated
    from ..core.config import load_config

    config = load_config("config.yaml")

    try:
        synth = BaseSynthesizer(config)
        print("  ✗ Should not be able to instantiate abstract class")
    except TypeError as e:
        print(f"  ✓ Abstract class cannot be instantiated: {e}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
