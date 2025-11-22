#!/usr/bin/env python3
"""Complete integration test - run with: python -m scripts.test_complete"""

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import _create_default_config
from src.core.logger import setup_root_logger, get_logger
from src.core.progress import ProgressManager
from src.providers.base import GenerationResult, FinishReason
from src.providers.load_balancer import LoadBalancer
from src.utils.cost_tracker import CostTracker
from src.utils.templates import TemplateManager
from src.utils.checkpoint import CheckpointManager, ResumableTask
from src.utils.output_handler import LocalOutputHandler
from src.utils.coordinator import InstanceCoordinator, WorkDistributor
from src.validators.language import LanguageValidator
from src.validators.quality import QualityScorer
from src.utils.json_parser import extract_json, parse_qa_variants


def test_complete_pipeline():
    """Test complete synthesis pipeline."""
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE TEST")
    print("=" * 60)

    # Setup
    tmpdir = tempfile.mkdtemp()
    config = _create_default_config()
    config.output.local_path = tmpdir

    try:
        # 1. Initialize components
        print("\n1. Initializing components...")

        progress = ProgressManager(config)
        print("   ✓ Progress manager")

        templates = TemplateManager()
        print("   ✓ Template manager")

        cost_tracker = CostTracker()
        print("   ✓ Cost tracker")

        checkpoint = CheckpointManager(os.path.join(tmpdir, 'checkpoints'))
        print("   ✓ Checkpoint manager")

        output = LocalOutputHandler(os.path.join(tmpdir, 'output'))
        print("   ✓ Output handler")

        language_val = LanguageValidator('id', 0.7)
        print("   ✓ Language validator")

        quality_scorer = QualityScorer()
        print("   ✓ Quality scorer")

        # 2. Generate prompts
        print("\n2. Generating prompts...")

        topics = ['Hukum Pidana', 'Hukum Perdata']
        for topic in topics:
            prompt = templates.render(
                'qa_synthesis',
                num_variants=3,
                topic=topic,
                language='Indonesian',
                min_length=100,
                max_length=500
            )
            print(f"   ✓ {topic}: {len(prompt)} chars")

        # 3. Simulate API response parsing
        print("\n3. Testing response parsing...")

        mock_response = '''```json
[
  {"question": "Apa itu hukum pidana?", "answer": "Hukum pidana adalah cabang hukum yang mengatur tindak pidana dan sanksinya."},
  {"question": "Siapa yang berwenang?", "answer": "Kepolisian, kejaksaan, dan pengadilan berwenang dalam proses hukum pidana."}
]
```'''

        json_content = extract_json(mock_response)
        variants = parse_qa_variants(json_content)
        print(f"   ✓ Parsed {len(variants)} QA pairs")

        # 4. Validate and score
        print("\n4. Validating content...")

        for i, variant in enumerate(variants):
            q, a = variant['question'], variant['answer']

            # Language check
            lang_result = language_val.validate_qa_pair(q, a)

            # Quality check
            item = {'question': q, 'answer': a}
            quality = quality_scorer.score(item, item)

            print(f"   ✓ QA {i+1}: quality={quality['overall_score']:.2f}")

        # 5. Track progress
        print("\n5. Tracking progress...")

        for topic in topics:
            progress.mark_topic_progress(topic, 10, 50)
        progress.add_generated(len(variants))
        progress.save(upload_to_hf=False)
        print(f"   ✓ Progress saved: {progress.progress_data['total_generated']} items")

        # 6. Track costs
        print("\n6. Tracking costs...")

        cost_tracker.record_usage('gemini', 'flash', 1000, 500)
        summary = cost_tracker.get_summary()
        print(f"   ✓ Cost tracked: ${summary['total_cost_usd']:.6f}")

        # 7. Save output
        print("\n7. Saving output...")

        data = [
            {'topic': 'Hukum Pidana', 'question': v['question'], 'answer': v['answer']}
            for v in variants
        ]
        path = output.save_batch(data, 'test_output')
        print(f"   ✓ Saved to: {path}")

        # 8. Checkpoint
        print("\n8. Testing checkpoint...")

        state = {
            'processed': len(variants),
            'topics': topics
        }
        checkpoint.save_checkpoint('test_run', state)
        loaded = checkpoint.load_checkpoint('latest')
        print(f"   ✓ Checkpoint saved and loaded: {loaded['state']['processed']} items")

        print("\n" + "=" * 60)
        print("✓ COMPLETE PIPELINE TEST PASSED")
        print("=" * 60)

        return True

    finally:
        shutil.rmtree(tmpdir)


def test_multi_instance():
    """Test multi-instance coordination."""
    print("\n" + "=" * 60)
    print("MULTI-INSTANCE TEST")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()

    try:
        # Create coordinator
        coord = InstanceCoordinator(lock_dir=tmpdir)
        print(f"   ✓ Instance: {coord.instance_id}")

        # Create distributor
        dist = WorkDistributor(coord)

        # Test item distribution
        items = [f'item_{i:03d}' for i in range(10)]

        # Acquire items
        acquired = dist.distribute_items(items, batch_size=3)
        print(f"   ✓ Acquired: {len(acquired)} items")

        # Mark some as completed
        for item in acquired[:2]:
            coord.mark_completed(item)
        print(f"   ✓ Completed: 2 items")

        # Get remaining
        remaining = [i for i in items if not coord.is_completed(i)]
        print(f"   ✓ Remaining: {len(remaining)} items")

        # Heartbeat
        coord.heartbeat({'processed': 2})
        print(f"   ✓ Heartbeat sent")

        # Get active instances
        active = coord.get_active_instances()
        print(f"   ✓ Active instances: {len(active)}")

        # Cleanup
        coord.deregister()
        print(f"   ✓ Deregistered")

        print("\n" + "=" * 60)
        print("✓ MULTI-INSTANCE TEST PASSED")
        print("=" * 60)

        return True

    finally:
        shutil.rmtree(tmpdir)


def test_resumable_task():
    """Test resumable task workflow."""
    print("\n" + "=" * 60)
    print("RESUMABLE TASK TEST")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()

    try:
        checkpoint = CheckpointManager(tmpdir)

        # Create task
        task = ResumableTask('synthesis', checkpoint)

        # Process items
        all_items = [f'item_{i:03d}' for i in range(20)]

        for item in all_items[:5]:
            task.mark_processed(item)

        task.update_statistics('generated', 50)
        task.save_checkpoint()
        print(f"   ✓ Processed: {task.processed_count} items")

        # Simulate restart
        task2 = ResumableTask('synthesis', checkpoint)
        task2.resume_from_checkpoint('latest')

        remaining = task2.get_remaining_items(all_items)
        print(f"   ✓ Resumed: {task2.processed_count} processed")
        print(f"   ✓ Remaining: {len(remaining)} items")

        assert task2.processed_count == 5
        assert len(remaining) == 15

        print("\n" + "=" * 60)
        print("✓ RESUMABLE TASK TEST PASSED")
        print("=" * 60)

        return True

    finally:
        shutil.rmtree(tmpdir)


def main():
    """Run complete integration tests."""
    setup_root_logger('WARNING')

    print("=" * 60)
    print("DATA SYNTHESIZER - COMPLETE INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        ("Complete Pipeline", test_complete_pipeline),
        ("Multi-Instance", test_multi_instance),
        ("Resumable Task", test_resumable_task),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
