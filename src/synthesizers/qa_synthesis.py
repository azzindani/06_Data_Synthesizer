"""QA pair synthesis engine."""

import os
import re
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

from .base import BaseSynthesizer
from ..core.config import Config
from ..core.logger import get_logger
from ..providers.factory import ProviderFactory


class QASynthesizer(BaseSynthesizer):
    """Synthesizer for generating QA pairs."""

    def __init__(self, config: Config, provider: ProviderFactory = None):
        """Initialize QA synthesizer.

        Args:
            config: Configuration object
            provider: Optional provider factory
        """
        super().__init__(config, provider)
        self.logger = get_logger(__name__)

        # Load domain configuration
        self._load_domain_config()

    def _load_domain_config(self):
        """Load domain-specific configuration."""
        # Default topics if not in config
        self.topics = self.config.domain.get('topics', [
            'General Knowledge',
            'Science and Technology',
            'History and Culture',
            'Business and Economics',
            'Health and Medicine'
        ])

        self.domain_name = self.config.domain.get('domain', 'General QA')
        self.target_language = self.config.quality.target_language
        self.num_variants = self.config.synthesis.num_variants

    def get_topics(self) -> List[str]:
        """Get list of topics to synthesize."""
        return self.topics

    def get_system_prompt(self) -> str:
        """Get system prompt for QA synthesis."""
        return f"""You are an expert in {self.domain_name}. Your task is to create high-quality question-answer pairs that help people understand various aspects of this domain. Generate educational, accurate, and comprehensive content."""

    def create_prompt(self, item: Dict) -> str:
        """Create synthesis prompt for a topic.

        Args:
            item: Dictionary with 'topic' key

        Returns:
            Prompt string
        """
        topic = item.get('topic', 'General')

        # Build JSON template
        json_format = []
        for i in range(self.num_variants):
            json_format.append(
                f'  {{"question": "question {i+1} about {topic}", '
                f'"answer": "comprehensive answer {i+1}"}}'
            )
        json_template = "[\n" + ",\n".join(json_format) + "\n]"

        return f"""Create {self.num_variants} high-quality question-answer pairs about: "{topic}"

REQUIREMENTS:
1. Each answer should be at least {self.config.quality.min_answer_length} words
2. Use accurate and up-to-date information
3. Include specific details, examples, or references where applicable
4. Each QA pair should take a different approach to the topic
5. Vary difficulty levels (basic to advanced)

VARIATION TYPES:
- Definitional: Basic concepts and terminology
- Procedural: How things work or are done
- Analytical: Analysis and comparison
- Practical: Real-world applications
- Advanced: Complex scenarios

FORMAT (JSON array only, no additional text):
{json_template}"""

    def _save_results(self, results: List[Dict], topic: str) -> None:
        """Save results to output.

        Args:
            results: Results to save
            topic: Topic name
        """
        if not results:
            return

        # Create DataFrame
        df = pd.DataFrame(results)

        # Clean topic name for filename
        clean_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"qa_{clean_topic}_{timestamp}.parquet"

        # Determine output path
        output_type = self.config.output.type

        if output_type in ['local', 'both']:
            # Save locally
            local_dir = self.config.output.local_path
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            df.to_parquet(local_path, index=False)
            self.logger.info(f"Saved locally: {local_path}")

        if output_type in ['huggingface', 'both']:
            # Upload to HuggingFace
            self._upload_to_hf(df, filename, topic, len(results))

    def _upload_to_hf(self, df: pd.DataFrame, filename: str, topic: str, count: int):
        """Upload results to HuggingFace.

        Args:
            df: DataFrame to upload
            filename: Filename for the upload
            topic: Topic name for commit message
            count: Number of items
        """
        try:
            from huggingface_hub import upload_file

            # Save to temp file
            temp_path = f"/tmp/{filename}"
            df.to_parquet(temp_path, index=False)

            # Upload
            upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=filename,
                repo_id=self.config.output.repository,
                repo_type="dataset",
                token=self.config.output.huggingface_token,
                commit_message=f"QA synthesis: {topic} - {count} pairs"
            )

            self.logger.info(f"Uploaded to HF: {filename}")

            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass

        except Exception as e:
            self.logger.error(f"Failed to upload to HF: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("QA SYNTHESIZER TEST")
    print("=" * 60)

    import os
    from ..core.config import _create_default_config

    # Create test config
    config = _create_default_config()
    config.output.type = "local"
    config.output.local_path = "./test_output"
    config.synthesis.num_variants = 3
    config.synthesis.questions_per_topic = 5

    # Set domain
    config.domain = {
        'domain': 'Test Domain',
        'topics': ['Test Topic 1', 'Test Topic 2']
    }

    # Set API key for testing
    if 'gemini' in config.providers:
        config.providers['gemini'].api_key = os.getenv("GEMINI_API_KEY", "")

    # Test 1: Create synthesizer
    try:
        synth = QASynthesizer(config)
        print("  ✓ QA Synthesizer created")
    except Exception as e:
        print(f"  ✗ Failed to create synthesizer: {e}")
        exit()

    # Test 2: Get topics
    topics = synth.get_topics()
    print(f"  ✓ Topics: {topics}")

    # Test 3: Get system prompt
    sys_prompt = synth.get_system_prompt()
    print(f"  ✓ System prompt: {len(sys_prompt)} chars")

    # Test 4: Create prompt
    prompt = synth.create_prompt({'topic': 'Test Topic'})
    print(f"  ✓ Created prompt: {len(prompt)} chars")

    # Test 5: Show progress
    synth.show_progress()

    # Test 6: Run synthesis (if API key available)
    if synth.provider.current_provider.api_key:
        print("\n  Running live synthesis test...")
        try:
            # Just test one item
            results = synth.synthesize_item({'topic': 'Basic Math'}, 'test_001')
            print(f"  ✓ Generated {len(results)} variants")
            for r in results:
                print(f"    - Q: {r['question'][:50]}...")
        except Exception as e:
            print(f"  ✗ Synthesis failed: {e}")
    else:
        print("  ⚠ No API key - skipping live synthesis test")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
