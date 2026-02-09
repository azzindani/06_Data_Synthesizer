#!/usr/bin/env python3
"""Test synthesis pipeline - run with: python -m scripts.test_synthesis"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import load_config
from src.core.logger import setup_root_logger
from src.core.progress import ProgressManager
from src.utils.json_parser import extract_json, parse_json_safely, parse_qa_variants
from src.utils.templates import TemplateManager
from src.validators.language import LanguageValidator
from src.validators.quality import QualityScorer


def test_qa_parsing():
    """Test QA response parsing."""
    print("\n" + "=" * 50)
    print("TEST: QA Response Parsing")
    print("=" * 50)

    # Simulate API response
    response = '''```json
[
  {
    "question": "Apa yang dimaksud dengan hukum pidana?",
    "answer": "Hukum pidana adalah cabang hukum yang mengatur tentang tindak pidana dan sanksinya. Hukum pidana bertujuan untuk melindungi kepentingan umum dan menjaga ketertiban masyarakat."
  },
  {
    "question": "Bagaimana proses peradilan pidana di Indonesia?",
    "answer": "Proses peradilan pidana di Indonesia dimulai dari penyidikan oleh kepolisian, penuntutan oleh jaksa, pemeriksaan di pengadilan, hingga putusan hakim. Setiap tahap memiliki aturan dan prosedur yang harus diikuti."
  }
]
```'''

    # Extract JSON
    json_content = extract_json(response)
    assert json_content is not None, "Should extract JSON"

    # Parse variants
    variants = parse_qa_variants(json_content)
    assert len(variants) == 2, f"Should have 2 variants, got {len(variants)}"

    # Check structure
    for v in variants:
        assert 'question' in v, "Should have question"
        assert 'answer' in v, "Should have answer"

    print(f"  ✓ Extracted JSON: {len(json_content)} chars")
    print(f"  ✓ Parsed {len(variants)} QA pairs")
    print(f"  ✓ Q1: {variants[0]['question'][:50]}...")

    return True


def test_prompt_generation():
    """Test prompt generation for different domains."""
    print("\n" + "=" * 50)
    print("TEST: Prompt Generation")
    print("=" * 50)

    tm = TemplateManager()

    # Indonesian Legal
    prompt_id = tm.render(
        'qa_synthesis',
        num_variants=5,
        topic='Hukum Pidana dan Sanksi',
        language='Indonesian',
        min_length=100,
        max_length=500
    )

    assert 'Hukum Pidana' in prompt_id
    print(f"  ✓ Indonesian Legal: {len(prompt_id)} chars")

    # Deep thinking
    prompt_dt = tm.render(
        'deep_thinking',
        topic='Contract Law',
        question='What are the requirements for a valid contract?',
        language='English'
    )

    assert 'reasoning' in prompt_dt.lower() or 'thinking' in prompt_dt.lower()
    print(f"  ✓ Deep Thinking: {len(prompt_dt)} chars")

    # Corpus generation
    prompt_corpus = tm.render(
        'corpus_generation',
        document_type='Service Agreement',
        industry='Technology',
        parties='Company A and Company B',
        purpose='Software development services',
        requirements='Payment terms, IP ownership',
        language='English'
    )

    print(f"  ✓ Corpus Generation: {len(prompt_corpus)} chars")

    return True


def test_quality_scoring():
    """Test quality scoring pipeline."""
    print("\n" + "=" * 50)
    print("TEST: Quality Scoring")
    print("=" * 50)

    qs = QualityScorer()

    # Good quality content
    good_q = "Bagaimana proses pembentukan undang-undang di Indonesia?"
    good_a = """Proses pembentukan undang-undang di Indonesia melibatkan beberapa tahapan penting.
    Pertama, rancangan undang-undang (RUU) dapat diajukan oleh Presiden atau DPR.
    Kedua, RUU dibahas bersama antara DPR dan Presiden melalui berbagai tingkat pembicaraan.
    Ketiga, setelah disetujui bersama, RUU disahkan oleh Presiden menjadi undang-undang.
    Proses ini diatur dalam UUD 1945 Pasal 20 dan UU No. 12 Tahun 2011 tentang Pembentukan Peraturan Perundang-undangan."""

    good_item = {'question': good_q, 'answer': good_a}
    good_scores = qs.score(good_item, good_item)

    print(f"  Good content scores:")
    print(f"    - Overall: {good_scores['overall_score']:.2f}")
    print(f"    - Length ratio: {good_scores['length_ratio']:.2f}")
    print(f"    - Vocabulary: {good_scores['vocabulary_richness']:.2f}")

    # Poor quality content
    poor_q = "Apa?"
    poor_a = "Ya."
    poor_item = {'question': poor_q, 'answer': poor_a}

    poor_scores = qs.score(poor_item, poor_item)

    print(f"  Poor content scores:")
    print(f"    - Overall: {poor_scores['overall_score']:.2f}")

    assert good_scores['overall_score'] > poor_scores['overall_score']
    print(f"  ✓ Good > Poor: {good_scores['overall_score']:.2f} > {poor_scores['overall_score']:.2f}")

    return True


def test_language_validation():
    """Test language validation."""
    print("\n" + "=" * 50)
    print("TEST: Language Validation")
    print("=" * 50)

    lv = LanguageValidator('id', 0.7)

    # Indonesian text
    id_text = "Hukum pidana adalah cabang hukum yang mengatur tentang tindak pidana dan sanksinya di Indonesia."
    id_result = lv.validate(id_text)

    print(f"  Indonesian text:")
    print(f"    - Language: {id_result.get('language', 'unknown')}")
    print(f"    - Confidence: {id_result.get('confidence', 0):.2f}")

    # QA pair validation
    qa_result = lv.validate_qa_pair(
        "Apa itu hukum pidana?",
        "Hukum pidana adalah cabang hukum publik yang mengatur perbuatan pidana."
    )

    print(f"  QA pair validation:")
    print(f"    - Overall valid: {qa_result.get('overall_valid', False)}")

    return True


def test_progress_tracking():
    """Test synthesis progress tracking."""
    print("\n" + "=" * 50)
    print("TEST: Progress Tracking")
    print("=" * 50)

    config = load_config("config.yaml")
    config.output.local_path = tempfile.mkdtemp()

    pm = ProgressManager(config)

    # Simulate synthesis progress
    topics = ['Hukum Pidana', 'Hukum Perdata', 'Hukum Bisnis']

    for i, topic in enumerate(topics):
        completed = (i + 1) * 10
        total = 50
        pm.mark_topic_progress(topic, completed, total)
        pm.add_generated(10)

    # Check progress
    stats = pm.get_statistics()

    print(f"  Topics tracked: {len(pm.progress_data['processed_topics'])}")
    print(f"  Total generated: {pm.progress_data['total_generated']}")

    # Save and verify
    pm.save(upload_to_hf=False)
    print(f"  ✓ Progress saved successfully")

    return True


def main():
    """Run all synthesis tests."""
    setup_root_logger('WARNING')

    print("=" * 60)
    print("DATA SYNTHESIZER - SYNTHESIS PIPELINE TESTS")
    print("=" * 60)

    tests = [
        ("QA Parsing", test_qa_parsing),
        ("Prompt Generation", test_prompt_generation),
        ("Quality Scoring", test_quality_scoring),
        ("Language Validation", test_language_validation),
        ("Progress Tracking", test_progress_tracking),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
