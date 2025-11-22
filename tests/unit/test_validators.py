"""Unit tests for validators."""

import pytest

from src.validators.language import LanguageValidator
from src.validators.quality import QualityScorer


class TestLanguageValidator:
    """Tests for language validator."""

    def test_create_validator(self):
        """Test creating validator."""
        validator = LanguageValidator(target_language="id")
        assert validator.target_language == "id"

    def test_validate_empty_text(self):
        """Test validation of empty text."""
        validator = LanguageValidator()
        result = validator.validate("")
        assert result['detected_language'] == 'unknown'
        assert result['is_valid'] == False

    def test_validate_short_text(self):
        """Test validation of short text."""
        validator = LanguageValidator()
        result = validator.validate("Hi")
        assert result['confidence'] == 0.0

    def test_validate_qa_pair(self):
        """Test QA pair validation."""
        validator = LanguageValidator(target_language="en")

        result = validator.validate_qa_pair(
            "What is programming?",
            "Programming is the process of writing computer code."
        )

        assert 'question_language' in result
        assert 'answer_language' in result
        assert 'both_valid' in result
        assert 'language_score' in result

    def test_validate_corpus(self):
        """Test corpus validation with sampling."""
        validator = LanguageValidator(target_language="en")

        long_text = "This is a test. " * 100
        result = validator.validate_corpus(long_text)

        assert 'detected_language' in result
        assert 'is_consistent' in result
        assert 'sample_count' in result


class TestQualityScorer:
    """Tests for quality scorer."""

    def test_create_scorer(self):
        """Test creating scorer."""
        scorer = QualityScorer()
        assert scorer is not None

    def test_score_good_content(self):
        """Test scoring good quality content."""
        scorer = QualityScorer()

        original = {
            'question': 'What is Python?',
            'answer': 'Python is a programming language.'
        }
        generated = {
            'question': 'Explain Python programming',
            'answer': 'Python is a high-level programming language known for its simplicity and readability. It is widely used in web development, data science, and automation.'
        }

        result = scorer.score(original, generated, 'en')

        assert 'overall_score' in result
        assert 'grade' in result
        assert 'recommendation' in result
        assert 0 <= result['overall_score'] <= 1

    def test_score_poor_content(self):
        """Test scoring poor quality content."""
        scorer = QualityScorer()

        original = {'question': 'Long question here', 'answer': 'Long answer here'}
        generated = {'question': 'Q', 'answer': 'A'}

        result = scorer.score(original, generated, 'en')

        assert result['overall_score'] < 0.5
        assert result['recommendation'] == 'reject'

    def test_grade_scores(self):
        """Test score grading."""
        scorer = QualityScorer()

        assert scorer._grade_score(0.90) == 'excellent'
        assert scorer._grade_score(0.75) == 'good'
        assert scorer._grade_score(0.60) == 'fair'
        assert scorer._grade_score(0.45) == 'poor'
        assert scorer._grade_score(0.30) == 'failing'

    def test_count_citations(self):
        """Test citation counting."""
        scorer = QualityScorer()

        text = "According to Pasal 1 and UU No. 5 Tahun 2020, as well as Article 10."
        count = scorer._count_citations(text)

        assert count >= 2

    def test_batch_scoring(self):
        """Test batch scoring."""
        scorer = QualityScorer()

        items = [
            {
                'original': {'question': 'Q1', 'answer': 'A1'},
                'generated': {'question': 'Question 1', 'answer': 'Answer 1 with more content here'}
            },
            {
                'original': {'question': 'Q2', 'answer': 'A2'},
                'generated': {'question': 'Question 2', 'answer': 'Answer 2 with more content here'}
            }
        ]

        stats = scorer.batch_score(items, 'en')

        assert stats['count'] == 2
        assert 'avg_score' in stats
        assert 'grades' in stats
        assert 'pass_rate' in stats

    def test_vocabulary_richness(self):
        """Test vocabulary richness scoring."""
        scorer = QualityScorer()

        # Good variety
        good_text = "The quick brown fox jumps over the lazy dog."
        good_score = scorer._score_vocabulary(good_text)

        # Repetitive
        bad_text = "test test test test test"
        bad_score = scorer._score_vocabulary(bad_text)

        assert good_score > bad_score

    def test_structure_scoring(self):
        """Test structure scoring."""
        scorer = QualityScorer()

        # Well structured
        good_text = """
        First paragraph here.

        Second paragraph here with more content.

        1. List item one
        2. List item two
        """

        # Poorly structured
        bad_text = "Just one long sentence without any structure or breaks"

        good_score = scorer._score_structure(good_text)
        bad_score = scorer._score_structure(bad_text)

        assert good_score > bad_score
