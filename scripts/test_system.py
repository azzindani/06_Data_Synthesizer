#!/usr/bin/env python3
"""Test all synthesizer components - run with: python -m scripts.test_system"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import _create_default_config, load_config
from src.core.logger import setup_root_logger, get_logger
from src.core.progress import ProgressManager
from src.providers.base import FinishReason, GenerationResult
from src.utils.json_parser import extract_json, parse_json_safely
from src.utils.cost_tracker import CostTracker
from src.utils.templates import TemplateManager
from src.utils.checkpoint import CheckpointManager
from src.validators.language import LanguageValidator
from src.validators.quality import QualityScorer


def test_config():
    """Test configuration system."""
    print("\n" + "=" * 50)
    print("TEST: Configuration System")
    print("=" * 50)

    config = _create_default_config()

    assert config.synthesis.questions_per_topic > 0, "Questions per topic should be > 0"
    assert config.retry.max_attempts > 0, "Max attempts should be > 0"
    assert len(config.providers) > 0, "Should have providers configured"

    print(f"  ✓ Default config loaded")
    print(f"  ✓ Questions per topic: {config.synthesis.questions_per_topic}")
    print(f"  ✓ Providers: {list(config.providers.keys())}")
    print(f"  ✓ Output type: {config.output.type}")

    return True


def test_progress_manager():
    """Test progress tracking."""
    print("\n" + "=" * 50)
    print("TEST: Progress Manager")
    print("=" * 50)

    import tempfile

    config = _create_default_config()
    config.output.local_path = tempfile.mkdtemp()

    pm = ProgressManager(config)

    # Test mark processed
    pm.mark_processed("item_001", {'quality': 'good'})
    pm.mark_processed("item_002")

    assert pm.is_processed("item_001"), "Item should be marked as processed"
    assert not pm.is_processed("item_999"), "Unknown item should not be processed"

    # Test topic progress
    pm.mark_topic_progress("Test Topic", 10, 50)

    # Test save
    pm.save(upload_to_hf=False)

    print(f"  ✓ Marked 2 items as processed")
    print(f"  ✓ Topic progress tracked")
    print(f"  ✓ Progress saved locally")

    return True


def test_json_parser():
    """Test JSON parsing utilities."""
    print("\n" + "=" * 50)
    print("TEST: JSON Parser")
    print("=" * 50)

    # Test extract from markdown
    text1 = '```json\n[{"question": "Q1", "answer": "A1"}]\n```'
    result1 = extract_json(text1)
    assert result1 is not None, "Should extract JSON from markdown"

    # Test parse safely
    parsed = parse_json_safely(result1)
    assert isinstance(parsed, list), "Should parse to list"
    assert parsed[0]['question'] == 'Q1', "Should have correct question"

    # Test with dirty JSON
    dirty = '{"key": "value",}'
    cleaned = parse_json_safely(dirty)
    assert cleaned is not None, "Should handle trailing comma"

    print(f"  ✓ Extract from markdown: {len(result1)} chars")
    print(f"  ✓ Parse safely: {len(parsed)} items")
    print(f"  ✓ Clean dirty JSON: success")

    return True


def test_cost_tracker():
    """Test cost tracking."""
    print("\n" + "=" * 50)
    print("TEST: Cost Tracker")
    print("=" * 50)

    tracker = CostTracker()

    # Record usage
    cost1 = tracker.record_usage("gemini", "flash", 1000, 500)
    cost2 = tracker.record_usage("openai", "gpt-4", 500, 200)

    # Get summary
    summary = tracker.get_summary()

    assert summary['total_requests'] == 2, "Should have 2 requests"
    assert summary['total_cost_usd'] > 0, "Should have cost"
    assert 'gemini' in summary['by_provider'], "Should track gemini"

    # Estimate remaining
    estimate = tracker.estimate_remaining_cost(100)

    print(f"  ✓ Recorded 2 API calls")
    print(f"  ✓ Total cost: ${summary['total_cost_usd']:.6f}")
    print(f"  ✓ Estimated 100 more: ${estimate:.4f}")

    return True


def test_templates():
    """Test template system."""
    print("\n" + "=" * 50)
    print("TEST: Template Manager")
    print("=" * 50)

    tm = TemplateManager()

    # List templates
    templates = tm.list_templates()
    assert len(templates) >= 3, "Should have at least 3 built-in templates"

    # Render template
    rendered = tm.render(
        'qa_synthesis',
        num_variants=5,
        topic='Hukum Pidana',
        language='Indonesian',
        min_length=100,
        max_length=500
    )

    assert '5' in rendered, "Should include num_variants"
    assert 'Hukum Pidana' in rendered, "Should include topic"

    # Create prompt
    system, user = tm.create_prompt(
        'qa_synthesis',
        'qa_expert',
        num_variants=3,
        topic='Test',
        language='id',
        min_length=50,
        max_length=200
    )

    print(f"  ✓ Templates available: {templates}")
    print(f"  ✓ Rendered prompt: {len(rendered)} chars")
    print(f"  ✓ System prompt: {len(system)} chars")

    return True


def test_checkpoint():
    """Test checkpoint system."""
    print("\n" + "=" * 50)
    print("TEST: Checkpoint Manager")
    print("=" * 50)

    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp()

    try:
        cm = CheckpointManager(checkpoint_dir=tmpdir)

        # Save checkpoint
        state = {'processed': 100, 'errors': []}
        path = cm.save_checkpoint('test_run', state, {'version': '1.0'})

        # Load checkpoint
        loaded = cm.load_checkpoint('latest')
        assert loaded['state'] == state, "State should match"

        # List checkpoints
        checkpoints = cm.list_checkpoints()
        assert len(checkpoints) == 1, "Should have 1 checkpoint"

        print(f"  ✓ Saved checkpoint: {path}")
        print(f"  ✓ Loaded checkpoint: {loaded['state']['processed']} items")
        print(f"  ✓ Listed checkpoints: {len(checkpoints)}")

    finally:
        shutil.rmtree(tmpdir)

    return True


def test_validators():
    """Test validation components."""
    print("\n" + "=" * 50)
    print("TEST: Validators")
    print("=" * 50)

    # Language validator
    lv = LanguageValidator('id', 0.8)
    result = lv.validate("Ini adalah teks bahasa Indonesia yang cukup panjang untuk dideteksi")

    print(f"  ✓ Language validator initialized")
    print(f"  ✓ Validation result: {result.get('language', 'unknown')}")

    # Quality scorer
    qs = QualityScorer()
    test_item = {
        'question': "Apa itu hukum pidana?",
        'answer': "Hukum pidana adalah cabang hukum yang mengatur tentang tindak pidana dan sanksinya. " * 5
    }
    scores = qs.score(test_item, test_item)

    assert 'overall_score' in scores, "Should have overall score"

    print(f"  ✓ Quality scorer initialized")
    print(f"  ✓ Quality score: {scores['overall_score']:.2f}")

    return True


def test_providers():
    """Test provider components."""
    print("\n" + "=" * 50)
    print("TEST: Providers")
    print("=" * 50)

    # Test GenerationResult
    result = GenerationResult(
        text="Test response",
        finish_reason=FinishReason.COMPLETE,
        tokens_used=100
    )

    assert result.success, "COMPLETE should be success"

    # Test finish reasons
    for reason in FinishReason:
        print(f"    - {reason.name}: {reason.value}")

    print(f"  ✓ GenerationResult created")
    print(f"  ✓ Success check: {result.success}")
    print(f"  ✓ All finish reasons available")

    return True


def main():
    """Run all tests."""
    setup_root_logger('WARNING')  # Quiet logging for tests

    print("=" * 60)
    print("DATA SYNTHESIZER - SYSTEM TESTS")
    print("=" * 60)

    tests = [
        ("Configuration", test_config),
        ("Progress Manager", test_progress_manager),
        ("JSON Parser", test_json_parser),
        ("Cost Tracker", test_cost_tracker),
        ("Templates", test_templates),
        ("Checkpoint", test_checkpoint),
        ("Validators", test_validators),
        ("Providers", test_providers),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {name} FAILED")
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
