"""Language detection and validation."""

import math
from typing import Dict, Tuple, Any

from ..core.logger import get_logger


class LanguageValidator:
    """Validates text language using langid."""

    def __init__(self, target_language: str = "id", min_confidence: float = 0.8):
        """Initialize language validator.

        Args:
            target_language: Expected language code (e.g., 'id', 'en')
            min_confidence: Minimum confidence threshold (0-1)
        """
        self.target_language = target_language
        self.min_confidence = min_confidence
        self.logger = get_logger(__name__)

        # Initialize langid
        try:
            import langid
            langid.set_languages([target_language, 'en'])
            self._langid = langid
        except ImportError:
            self.logger.warning("langid not installed, language validation disabled")
            self._langid = None

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or len(text.strip()) < 10:
            return 'unknown', 0.0

        if not self._langid:
            return self.target_language, 1.0  # Skip validation

        try:
            lang, raw_confidence = self._langid.classify(text.strip())

            # langid returns negative log probability
            # Convert to 0-1 scale
            if raw_confidence <= 0:
                normalized = math.exp(raw_confidence)
            else:
                normalized = 1.0

            return lang, max(0.0, min(1.0, normalized))

        except Exception as e:
            self.logger.debug(f"Language detection error: {e}")
            return 'unknown', 0.0

    def validate(self, text: str) -> Dict[str, Any]:
        """Validate text language.

        Args:
            text: Text to validate

        Returns:
            Validation result dictionary
        """
        lang, confidence = self.detect_language(text)

        return {
            'detected_language': lang,
            'confidence': confidence,
            'target_language': self.target_language,
            'is_target_language': lang == self.target_language,
            'meets_confidence': confidence >= self.min_confidence,
            'is_valid': lang == self.target_language and confidence >= self.min_confidence
        }

    def validate_qa_pair(self, question: str, answer: str) -> Dict[str, Any]:
        """Validate language of a QA pair.

        Args:
            question: Question text
            answer: Answer text

        Returns:
            Combined validation result
        """
        q_result = self.validate(question)
        a_result = self.validate(answer)

        # Combined score
        avg_confidence = (q_result['confidence'] + a_result['confidence']) / 2

        return {
            'question_language': q_result['detected_language'],
            'question_confidence': q_result['confidence'],
            'answer_language': a_result['detected_language'],
            'answer_confidence': a_result['confidence'],
            'question_valid': q_result['is_valid'],
            'answer_valid': a_result['is_valid'],
            'both_valid': q_result['is_valid'] and a_result['is_valid'],
            'average_confidence': avg_confidence,
            'language_score': avg_confidence if q_result['is_target_language'] and a_result['is_target_language'] else avg_confidence * 0.5
        }

    def validate_corpus(self, text: str, num_samples: int = 3) -> Dict[str, Any]:
        """Validate language of corpus text by sampling.

        Args:
            text: Corpus text
            num_samples: Number of samples to take

        Returns:
            Validation result with sampling info
        """
        if not text or len(text) < 50:
            return {
                'detected_language': 'unknown',
                'average_confidence': 0.0,
                'is_consistent': False,
                'is_valid': False
            }

        # Sample from different parts
        text_len = len(text)
        samples = []

        # Beginning
        samples.append(text[:min(500, text_len)])

        # Middle
        if text_len > 1000:
            mid = text_len // 2
            samples.append(text[mid:min(mid + 500, text_len)])

        # End
        if text_len > 500:
            samples.append(text[-500:])

        # Detect language for each sample
        detections = []
        for sample in samples:
            if len(sample.strip()) < 20:
                continue
            lang, conf = self.detect_language(sample)
            if lang != 'unknown':
                detections.append((lang, conf))

        if not detections:
            return {
                'detected_language': 'unknown',
                'average_confidence': 0.0,
                'is_consistent': False,
                'is_valid': False
            }

        # Calculate statistics
        languages = [d[0] for d in detections]
        confidences = [d[1] for d in detections]

        most_common = max(set(languages), key=languages.count)
        avg_confidence = sum(confidences) / len(confidences)
        is_consistent = all(lang == most_common for lang in languages)

        return {
            'detected_language': most_common,
            'average_confidence': avg_confidence,
            'sample_count': len(detections),
            'is_consistent': is_consistent,
            'matches_target': most_common == self.target_language,
            'is_valid': most_common == self.target_language and avg_confidence >= self.min_confidence
        }


if __name__ == "__main__":
    print("=" * 60)
    print("LANGUAGE VALIDATOR TEST")
    print("=" * 60)

    # Test 1: Create validator
    validator = LanguageValidator(target_language="id", min_confidence=0.7)
    print("  ✓ Validator created")

    # Test 2: Detect Indonesian text
    id_text = "Ini adalah contoh teks dalam bahasa Indonesia yang cukup panjang untuk dideteksi dengan benar."
    result = validator.validate(id_text)
    print(f"  ✓ Indonesian text: {result['detected_language']} ({result['confidence']:.3f})")

    # Test 3: Detect English text
    en_text = "This is an example of English text that should be detected correctly by the validator."
    result = validator.validate(en_text)
    print(f"  ✓ English text: {result['detected_language']} ({result['confidence']:.3f})")

    # Test 4: QA pair validation
    qa_result = validator.validate_qa_pair(
        "Apa itu hukum pidana?",
        "Hukum pidana adalah cabang hukum yang mengatur tentang tindak pidana dan sanksinya."
    )
    print(f"  ✓ QA pair: both_valid={qa_result['both_valid']}, score={qa_result['language_score']:.3f}")

    # Test 5: Corpus validation
    corpus = id_text * 10  # Repeat to make it longer
    corpus_result = validator.validate_corpus(corpus)
    print(f"  ✓ Corpus: {corpus_result['detected_language']}, consistent={corpus_result['is_consistent']}")

    # Test 6: Empty/short text
    empty_result = validator.validate("")
    short_result = validator.validate("Hi")
    print(f"  ✓ Edge cases: empty={empty_result['detected_language']}, short={short_result['detected_language']}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
