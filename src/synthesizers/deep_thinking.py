"""Deep thinking QA synthesis with reasoning traces."""

import os
import re
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

from .base import BaseSynthesizer
from ..core.config import Config
from ..core.logger import get_logger
from ..providers.factory import ProviderFactory
from ..utils.json_parser import parse_json_safely


class DeepThinkingSynthesizer(BaseSynthesizer):
    """Synthesizer for QA pairs with detailed thinking/reasoning traces."""

    def __init__(self, config: Config, provider: ProviderFactory = None):
        """Initialize deep thinking synthesizer.

        Args:
            config: Configuration object
            provider: Optional provider factory
        """
        super().__init__(config, provider)
        self.logger = get_logger(__name__)
        self._load_domain_config()

    def _load_domain_config(self):
        """Load domain-specific configuration."""
        self.topics = self.config.domain.get('topics', [
            'Complex Problem Analysis',
            'Ethical Dilemmas',
            'Strategic Decision Making',
            'Policy Evaluation',
            'Case Study Analysis'
        ])

        self.domain_name = self.config.domain.get('domain', 'Deep Analysis')
        self.target_language = self.config.quality.target_language
        self.num_variants = self.config.synthesis.num_variants

        # Deep thinking specific settings
        self.min_thinking_words = self.config.domain.get('min_thinking_words', 500)
        self.max_answer_words = self.config.domain.get('max_answer_words', 200)

    def get_topics(self) -> List[str]:
        """Get list of topics to synthesize."""
        return self.topics

    def get_system_prompt(self) -> str:
        """Get system prompt for deep thinking synthesis."""
        return f"""You are an expert analyst in {self.domain_name} with exceptional analytical capabilities. Your task is to perform iterative deep thinking analysis, showing your complete reasoning process with multiple iterations of reflection and fact-checking, then provide a concise final answer.

Your thinking process must demonstrate:
- Initial analysis and hypothesis formation
- Deep exploration of relevant factors
- Critical evaluation and reconsideration
- Multiple iterations of refinement
- Final synthesis and conclusion"""

    def create_prompt(self, item: Dict) -> str:
        """Create synthesis prompt for deep thinking.

        Args:
            item: Dictionary with 'topic' key

        Returns:
            Prompt string
        """
        topic = item.get('topic', 'General Analysis')

        # Build JSON template
        json_format = []
        for i in range(self.num_variants):
            json_format.append(
                f'  {{"question": "complex question {i+1} about {topic}", '
                f'"thinking": "detailed iterative thinking process {i+1}", '
                f'"answer": "concise final answer {i+1}"}}'
            )
        json_template = "[\n" + ",\n".join(json_format) + "\n]"

        return f"""Create {self.num_variants} question-answer pairs with ITERATIVE DEEP THINKING about: "{topic}"

REQUIREMENTS FOR THINKING PROCESS:
1. Thinking must be very detailed (minimum {self.min_thinking_words} words)
2. Show at least 3 iterations of analysis and reconsideration
3. Include fact-checking and source verification steps
4. Demonstrate perspective shifts and critical evaluation
5. Use phrases like: "upon further reflection", "reconsidering this", "after verification"

REQUIREMENTS FOR ANSWER:
1. Answer must be concise (maximum {self.max_answer_words} words)
2. Directly address the question
3. Synthesize the thinking into clear conclusions

THINKING STRUCTURE TO FOLLOW:
1. INITIAL ANALYSIS: First understanding and hypothesis
2. DEEP EXPLORATION: Research and evidence gathering
3. CRITICAL EVALUATION: Challenge assumptions
4. ITERATION 1: Revise based on new insights
5. FACT-CHECK: Verify claims and sources
6. ITERATION 2: Further refinement
7. ALTERNATIVE PERSPECTIVES: Consider other viewpoints
8. ITERATION 3: Final synthesis
9. CONCLUSION: Consolidated understanding

TARGET RATIO: Thinking should be 4-8x longer than Answer

FORMAT (JSON array only):
{json_template}"""

    def _process_variants(self, variants: List[Dict], original: Dict, item_id: str) -> List[Dict]:
        """Process variants with thinking quality assessment.

        Args:
            variants: Raw variants from LLM
            original: Original item
            item_id: Item identifier

        Returns:
            Processed results
        """
        results = []
        quality_config = self.config.quality

        for i, variant in enumerate(variants):
            question = variant.get('question', '').strip()
            thinking = variant.get('thinking', '').strip()
            answer = variant.get('answer', '').strip()

            # Basic validation
            if len(question) < 10 or len(thinking) < 100 or len(answer) < 20:
                continue

            # Calculate metrics
            thinking_words = len(thinking.split())
            answer_words = len(answer.split())
            ratio = thinking_words / answer_words if answer_words > 0 else 0

            # Assess thinking quality
            thinking_quality = self._assess_thinking_quality(thinking)

            result = {
                'item_id': item_id,
                'variant_number': i + 1,
                'question': question,
                'thinking': thinking,
                'answer': answer,
                'question_length': len(question),
                'thinking_length': len(thinking),
                'answer_length': len(answer),
                'thinking_word_count': thinking_words,
                'answer_word_count': answer_words,
                'thinking_answer_ratio': ratio,
                'meets_thinking_target': thinking_words >= self.min_thinking_words,
                'meets_answer_target': answer_words <= self.max_answer_words,
                'optimal_structure': thinking_words >= self.min_thinking_words and answer_words <= self.max_answer_words,
                **thinking_quality,
                'timestamp': datetime.now().isoformat(),
                'provider': self.provider.current_provider_name
            }

            results.append(result)

        return results

    def _assess_thinking_quality(self, thinking: str) -> Dict[str, Any]:
        """Assess quality of thinking process.

        Args:
            thinking: Thinking text

        Returns:
            Quality metrics
        """
        thinking_lower = thinking.lower()

        # Iteration indicators
        iteration_phrases = [
            'setelah mempertimbangkan', 'upon reflection', 'reconsidering',
            'revisi', 'iteration', 'further analysis', 'looking again',
            'dengan mempertimbangkan kembali', 'setelah meneliti lebih lanjut'
        ]
        iteration_count = sum(1 for phrase in iteration_phrases if phrase in thinking_lower)

        # Fact-check indicators
        fact_phrases = [
            'verifikasi', 'verification', 'fact-check', 'confirm',
            'validasi', 'cross-check', 'evidence shows'
        ]
        fact_count = sum(1 for phrase in fact_phrases if phrase in thinking_lower)

        # Perspective indicators
        perspective_phrases = [
            'di sisi lain', 'on the other hand', 'alternatively',
            'from another perspective', 'however', 'namun', 'sebaliknya'
        ]
        perspective_count = sum(1 for phrase in perspective_phrases if phrase in thinking_lower)

        # Total depth score
        total_score = iteration_count + fact_count + perspective_count

        # Grade
        if total_score >= 10 and iteration_count >= 3:
            grade = 'excellent'
        elif total_score >= 6 and iteration_count >= 2:
            grade = 'good'
        elif total_score >= 3:
            grade = 'fair'
        else:
            grade = 'poor'

        return {
            'iteration_count': iteration_count,
            'fact_check_count': fact_count,
            'perspective_count': perspective_count,
            'depth_score': total_score,
            'thinking_grade': grade,
            'has_iterations': iteration_count >= 2,
            'has_fact_checks': fact_count >= 1,
            'has_perspectives': perspective_count >= 2
        }

    def _save_results(self, results: List[Dict], topic: str) -> None:
        """Save deep thinking results."""
        if not results:
            return

        df = pd.DataFrame(results)

        clean_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"deep_thinking_{clean_topic}_{timestamp}.parquet"

        output_type = self.config.output.type

        if output_type in ['local', 'both']:
            local_dir = self.config.output.local_path
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            df.to_parquet(local_path, index=False)
            self.logger.info(f"Saved locally: {local_path}")

        if output_type in ['huggingface', 'both']:
            self._upload_to_hf(df, filename, topic, len(results))

    def _upload_to_hf(self, df: pd.DataFrame, filename: str, topic: str, count: int):
        """Upload to HuggingFace."""
        try:
            from huggingface_hub import upload_file

            temp_path = f"/tmp/{filename}"
            df.to_parquet(temp_path, index=False)

            upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=filename,
                repo_id=self.config.output.repository,
                repo_type="dataset",
                token=self.config.output.huggingface_token,
                commit_message=f"Deep thinking: {topic} - {count} pairs"
            )

            self.logger.info(f"Uploaded to HF: {filename}")
            os.remove(temp_path)

        except Exception as e:
            self.logger.error(f"Failed to upload: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("DEEP THINKING SYNTHESIZER TEST")
    print("=" * 60)

    from ..core.config import _create_default_config
    import os

    config = _create_default_config()
    config.output.type = "local"
    config.output.local_path = "./test_output"
    config.synthesis.num_variants = 2

    config.domain = {
        'domain': 'Legal Analysis',
        'topics': ['Contract Disputes', 'Criminal Law Cases'],
        'min_thinking_words': 300,
        'max_answer_words': 150
    }

    if 'gemini' in config.providers:
        config.providers['gemini'].api_key = os.getenv("GEMINI_API_KEY", "")

    try:
        synth = DeepThinkingSynthesizer(config)
        print("  ✓ Deep Thinking Synthesizer created")

        topics = synth.get_topics()
        print(f"  ✓ Topics: {topics}")

        prompt = synth.create_prompt({'topic': 'Test Topic'})
        print(f"  ✓ Prompt length: {len(prompt)} chars")

        # Test thinking quality assessment
        sample_thinking = """
        Upon initial analysis, this case presents several complex issues.
        After further reflection and reviewing the evidence, I need to reconsider
        my initial position. The fact-check of sources confirms that...
        On the other hand, we must consider alternative perspectives.
        Through this iteration of analysis, a clearer picture emerges.
        """
        quality = synth._assess_thinking_quality(sample_thinking)
        print(f"  ✓ Thinking quality: {quality['thinking_grade']}, depth={quality['depth_score']}")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
