"""Quality scoring for generated content."""

import re
import hashlib
from typing import Dict, Any, List

from ..core.logger import get_logger


class QualityScorer:
    """Scores quality of generated content."""

    def __init__(self, config: Any = None):
        """Initialize quality scorer.

        Args:
            config: Optional configuration object
        """
        self.logger = get_logger(__name__)
        self.config = config

        # Scoring weights
        self.weights = {
            'length_ratio': 0.15,
            'vocabulary_richness': 0.15,
            'structure_score': 0.20,
            'content_density': 0.20,
            'relevance_score': 0.30
        }

        # Legal terms for domain-specific scoring
        self.legal_terms = {
            'id': [
                'pasal', 'undang-undang', 'peraturan', 'hukum', 'pidana', 'perdata',
                'pengadilan', 'hakim', 'jaksa', 'terdakwa', 'tergugat', 'penggugat',
                'putusan', 'vonis', 'banding', 'kasasi', 'mahkamah'
            ],
            'en': [
                'article', 'law', 'regulation', 'statute', 'criminal', 'civil',
                'court', 'judge', 'prosecutor', 'defendant', 'plaintiff',
                'verdict', 'appeal', 'jurisdiction', 'litigation'
            ]
        }

        # Citation patterns
        self.citation_patterns = [
            r'Pasal\s+\d+',
            r'UU\s+(No\.|Nomor)\s*\d+',
            r'PP\s+(No\.|Nomor)\s*\d+',
            r'Peraturan\s+\w+\s+Nomor\s*\d+',
            r'Article\s+\d+',
            r'Section\s+\d+',
        ]

    def score(self, original: Dict, generated: Dict, language: str = 'id') -> Dict[str, Any]:
        """Score generated content quality.

        Args:
            original: Original content dict with 'question' and 'answer'
            generated: Generated content dict
            language: Language code for domain terms

        Returns:
            Quality scores dictionary
        """
        orig_q = str(original.get('question', ''))
        orig_a = str(original.get('answer', ''))
        gen_q = str(generated.get('question', ''))
        gen_a = str(generated.get('answer', ''))

        # Calculate individual scores
        scores = {
            'length_ratio': self._score_length_ratio(orig_a, gen_a),
            'vocabulary_richness': self._score_vocabulary(gen_a),
            'structure_score': self._score_structure(gen_a),
            'content_density': self._score_content_density(gen_a, language),
            'relevance_score': self._score_relevance(orig_q + orig_a, gen_q + gen_a)
        }

        # Calculate weighted overall score
        overall = sum(scores[k] * self.weights[k] for k in self.weights)
        scores['overall_score'] = overall

        # Quality grade
        scores['grade'] = self._grade_score(overall)

        # Additional metrics
        scores['word_count'] = len(gen_a.split())
        scores['char_count'] = len(gen_a)
        scores['citation_count'] = self._count_citations(gen_a)
        scores['has_citations'] = scores['citation_count'] > 0

        # Recommendation
        if overall >= 0.75:
            scores['recommendation'] = 'keep'
        elif overall >= 0.5:
            scores['recommendation'] = 'review'
        else:
            scores['recommendation'] = 'reject'

        return scores

    def _score_length_ratio(self, original: str, generated: str) -> float:
        """Score based on length ratio."""
        orig_len = len(original.split())
        gen_len = len(generated.split())

        if orig_len == 0:
            return 1.0 if gen_len >= 50 else 0.5

        ratio = gen_len / orig_len

        # Ideal ratio is around 0.8-1.2
        if 0.8 <= ratio <= 1.2:
            return 1.0
        elif 0.5 <= ratio <= 1.5:
            return 0.8
        elif 0.3 <= ratio <= 2.0:
            return 0.5
        else:
            return 0.3

    def _score_vocabulary(self, text: str) -> float:
        """Score vocabulary richness."""
        words = text.lower().split()
        if len(words) == 0:
            return 0.0

        unique = set(words)
        richness = len(unique) / len(words)

        # Ideal richness is 0.4-0.7 (some repetition is normal)
        if 0.4 <= richness <= 0.7:
            return 1.0
        elif 0.3 <= richness <= 0.8:
            return 0.8
        elif richness < 0.3:
            return 0.4  # Too repetitive
        else:
            return 0.6  # Too random

    def _score_structure(self, text: str) -> float:
        """Score text structure."""
        score = 0.0

        # Has paragraphs
        if '\n\n' in text or text.count('\n') >= 2:
            score += 0.3

        # Has sentences (periods)
        sentences = text.count('.') + text.count('。')
        if sentences >= 3:
            score += 0.3

        # Has lists or numbering
        if re.search(r'^\s*[\d\-\*]\s*', text, re.MULTILINE):
            score += 0.2

        # Reasonable sentence length (not too long)
        words = len(text.split())
        if sentences > 0:
            avg_sentence = words / sentences
            if 10 <= avg_sentence <= 30:
                score += 0.2
            elif avg_sentence < 50:
                score += 0.1

        return min(1.0, score)

    def _score_content_density(self, text: str, language: str) -> float:
        """Score content density based on domain terms."""
        text_lower = text.lower()
        words = text_lower.split()

        if len(words) == 0:
            return 0.0

        # Count domain terms
        terms = self.legal_terms.get(language, self.legal_terms['en'])
        term_count = sum(1 for word in words if any(term in word for term in terms))

        # Density ratio
        density = term_count / len(words)

        # Ideal density is 0.05-0.15 (5-15% domain terms)
        if 0.05 <= density <= 0.15:
            return 1.0
        elif 0.02 <= density <= 0.20:
            return 0.8
        elif density < 0.02:
            return 0.4  # Too generic
        else:
            return 0.6  # Too jargon-heavy

    def _score_relevance(self, original: str, generated: str) -> float:
        """Score relevance using simple term overlap."""
        orig_words = set(original.lower().split())
        gen_words = set(generated.lower().split())

        if len(orig_words) == 0:
            return 0.5

        # Calculate Jaccard similarity
        intersection = orig_words.intersection(gen_words)
        union = orig_words.union(gen_words)

        if len(union) == 0:
            return 0.0

        similarity = len(intersection) / len(union)

        # Scale to quality score (some overlap is good, too much is copying)
        if 0.2 <= similarity <= 0.5:
            return 1.0
        elif 0.1 <= similarity <= 0.6:
            return 0.8
        elif similarity > 0.6:
            return 0.5  # Too similar (possibly copied)
        else:
            return 0.4  # Too different

    def _count_citations(self, text: str) -> int:
        """Count legal citations in text."""
        count = 0
        for pattern in self.citation_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def _grade_score(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.85:
            return 'excellent'
        elif score >= 0.70:
            return 'good'
        elif score >= 0.55:
            return 'fair'
        elif score >= 0.40:
            return 'poor'
        else:
            return 'failing'

    def batch_score(self, items: List[Dict], language: str = 'id') -> Dict[str, Any]:
        """Score a batch of items and return statistics.

        Args:
            items: List of dicts with 'original' and 'generated' keys
            language: Language code

        Returns:
            Batch statistics
        """
        if not items:
            return {'count': 0, 'avg_score': 0.0}

        scores = []
        grades = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'failing': 0}

        for item in items:
            result = self.score(
                item.get('original', {}),
                item.get('generated', {}),
                language
            )
            scores.append(result['overall_score'])
            grades[result['grade']] += 1

        return {
            'count': len(items),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'grades': grades,
            'pass_rate': (grades['excellent'] + grades['good'] + grades['fair']) / len(items)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("QUALITY SCORER TEST")
    print("=" * 60)

    scorer = QualityScorer()

    # Test 1: Score good quality content
    original = {
        'question': 'Apa itu hukum pidana?',
        'answer': 'Hukum pidana adalah cabang hukum yang mengatur tentang tindak pidana dan sanksinya berdasarkan Pasal 1 KUHP.'
    }
    generated = {
        'question': 'Jelaskan pengertian hukum pidana',
        'answer': 'Hukum pidana merupakan bagian dari sistem hukum yang mengatur perbuatan yang dilarang oleh undang-undang. Berdasarkan Pasal 1 KUHP, hukum pidana mencakup ketentuan tentang tindak pidana dan pemidanaan. Sanksi dalam hukum pidana dapat berupa pidana penjara, denda, atau hukuman lainnya sesuai dengan ketentuan yang berlaku.'
    }

    result = scorer.score(original, generated, 'id')
    print(f"  ✓ Good content score: {result['overall_score']:.3f} ({result['grade']})")
    print(f"    - Length ratio: {result['length_ratio']:.2f}")
    print(f"    - Vocabulary: {result['vocabulary_richness']:.2f}")
    print(f"    - Structure: {result['structure_score']:.2f}")
    print(f"    - Content density: {result['content_density']:.2f}")
    print(f"    - Relevance: {result['relevance_score']:.2f}")

    # Test 2: Score poor quality (too short)
    poor_generated = {
        'question': 'Apa hukum?',
        'answer': 'Hukum adalah aturan.'
    }
    result = scorer.score(original, poor_generated, 'id')
    print(f"  ✓ Poor content score: {result['overall_score']:.3f} ({result['grade']})")

    # Test 3: Count citations
    text_with_citations = "Berdasarkan Pasal 338 KUHP dan UU No. 1 Tahun 2023 tentang KUHP baru."
    count = scorer._count_citations(text_with_citations)
    print(f"  ✓ Citation count: {count}")

    # Test 4: Batch scoring
    batch = [
        {'original': original, 'generated': generated},
        {'original': original, 'generated': poor_generated}
    ]
    stats = scorer.batch_score(batch, 'id')
    print(f"  ✓ Batch stats: avg={stats['avg_score']:.3f}, pass_rate={stats['pass_rate']:.1%}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
