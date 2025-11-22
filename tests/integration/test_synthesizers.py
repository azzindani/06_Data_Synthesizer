"""Integration tests for synthesizers."""

import pytest
import os

from src.core.config import _create_default_config
from src.synthesizers.qa_synthesis import QASynthesizer
from src.synthesizers.deep_thinking import DeepThinkingSynthesizer
from src.synthesizers.corpus_generator import CorpusGenerator


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set"
)


@pytest.fixture
def config():
    """Create test configuration."""
    config = _create_default_config()
    config.output.type = "local"
    config.output.local_path = "./test_output"
    config.synthesis.num_variants = 2
    config.synthesis.questions_per_topic = 2

    if 'gemini' in config.providers:
        config.providers['gemini'].api_key = os.getenv("GEMINI_API_KEY", "")

    config.domain = {
        'domain': 'Test Domain',
        'topics': ['Test Topic']
    }

    return config


class TestQASynthesizer:
    """Integration tests for QA synthesizer."""

    @pytest.mark.integration
    def test_synthesize_single_item(self, config):
        """Test generating QA pairs for a single item."""
        synth = QASynthesizer(config)

        results = synth.synthesize_item(
            {'topic': 'Basic Math'},
            'test_qa_001'
        )

        assert len(results) > 0
        assert 'question' in results[0]
        assert 'answer' in results[0]
        assert len(results[0]['question']) > 10
        assert len(results[0]['answer']) > 50

    @pytest.mark.integration
    def test_prompt_generation(self, config):
        """Test prompt creation."""
        synth = QASynthesizer(config)

        prompt = synth.create_prompt({'topic': 'Test'})

        assert 'Test' in prompt
        assert 'JSON' in prompt
        assert str(config.synthesis.num_variants) in prompt


class TestDeepThinkingSynthesizer:
    """Integration tests for deep thinking synthesizer."""

    @pytest.mark.integration
    def test_synthesize_with_thinking(self, config):
        """Test generating QA with thinking traces."""
        config.domain['min_thinking_words'] = 100
        config.domain['max_answer_words'] = 100

        synth = DeepThinkingSynthesizer(config)

        results = synth.synthesize_item(
            {'topic': 'Problem Analysis'},
            'test_deep_001'
        )

        assert len(results) > 0
        assert 'thinking' in results[0]
        assert 'answer' in results[0]
        assert results[0]['thinking_word_count'] > 50

    @pytest.mark.integration
    def test_thinking_quality_assessment(self, config):
        """Test thinking quality assessment."""
        synth = DeepThinkingSynthesizer(config)

        sample = """
        Upon initial analysis, this problem requires careful consideration.
        After further reflection and reviewing the evidence, I reconsidered
        my approach. The fact-check of sources confirms that my initial
        hypothesis was partially correct. However, on the other hand,
        we must consider alternative perspectives.
        """

        quality = synth._assess_thinking_quality(sample)

        assert quality['depth_score'] > 0
        assert quality['thinking_grade'] in ['excellent', 'good', 'fair', 'poor']


class TestCorpusGenerator:
    """Integration tests for corpus generator."""

    @pytest.mark.integration
    def test_generate_document(self, config):
        """Test generating a full document."""
        config.domain = {
            'domain': 'Legal Documents',
            'min_document_length': 200,
            'taxonomy': {
                'document_types': ['Simple Agreement'],
                'industries': ['Technology'],
                'jurisdictions': ['US Law']
            }
        }

        gen = CorpusGenerator(config)

        results = gen.synthesize_item(
            {'topic': 'Simple Agreement'},
            'test_corpus_001'
        )

        assert len(results) > 0
        assert 'document_text' in results[0]
        assert results[0]['word_count'] > 100


class TestProviderFailover:
    """Integration tests for provider failover."""

    @pytest.mark.integration
    def test_provider_usage_tracking(self, config):
        """Test that provider usage is tracked."""
        synth = QASynthesizer(config)

        # Generate something
        synth.synthesize_item({'topic': 'Test'}, 'test_usage_001')

        usage = synth.provider.get_usage()

        assert usage['total_requests'] > 0
