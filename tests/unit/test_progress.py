"""Unit tests for progress manager."""

import pytest
import json
import os

from src.core.progress import ProgressManager


class TestProgressManager:
    """Tests for ProgressManager class."""

    def test_create_without_config(self):
        """Test creating progress manager without config."""
        pm = ProgressManager()
        assert pm.progress_file == "synthesis_progress.json"
        assert pm.progress_data['total_processed'] == 0

    def test_mark_processed(self):
        """Test marking items as processed."""
        pm = ProgressManager()

        pm.mark_processed("item_001")
        pm.mark_processed("item_002")

        assert pm.progress_data['total_processed'] == 2
        assert "item_001" in pm.progress_data['processed_items']
        assert "item_002" in pm.progress_data['processed_items']

    def test_mark_processed_with_stats(self):
        """Test marking items with statistics."""
        pm = ProgressManager()

        stats = {
            'quality_counts': {'good': 2, 'excellent': 1},
            'language_passed': 3
        }
        pm.mark_processed("item_001", stats)

        assert pm.progress_data['statistics']['quality_counts']['good'] == 2
        assert pm.progress_data['statistics']['language_accuracy']['passed'] == 3

    def test_is_processed(self):
        """Test checking if item is processed."""
        pm = ProgressManager()

        pm.mark_processed("item_001")

        assert pm.is_processed("item_001") == True
        assert pm.is_processed("item_002") == False

    def test_mark_topic_progress(self):
        """Test updating topic progress."""
        pm = ProgressManager()

        pm.mark_topic_progress("Topic A", 10, 100)

        assert "Topic A" in pm.progress_data['processed_topics']
        assert pm.progress_data['processed_topics']['Topic A']['completed'] == 10
        assert pm.progress_data['processed_topics']['Topic A']['percentage'] == 10.0

    def test_add_generated(self):
        """Test adding to generated count."""
        pm = ProgressManager()

        pm.add_generated(5)
        pm.add_generated(3)

        assert pm.progress_data['total_generated'] == 8

    def test_increment_requests(self):
        """Test incrementing request count."""
        pm = ProgressManager()

        count1 = pm.increment_requests()
        count2 = pm.increment_requests()

        assert count1 == 1
        assert count2 == 2
        assert pm.progress_data['request_count'] == 2

    def test_add_error(self):
        """Test adding errors."""
        pm = ProgressManager()

        pm.add_error({'type': 'test_error', 'message': 'Test'})

        assert len(pm.progress_data['errors']) == 1
        assert pm.progress_data['errors'][0]['type'] == 'test_error'

    def test_error_limit(self):
        """Test error list is limited to 100."""
        pm = ProgressManager()

        for i in range(150):
            pm.add_error({'type': f'error_{i}'})

        assert len(pm.progress_data['errors']) == 100

    def test_get_resume_point(self):
        """Test getting resume point."""
        pm = ProgressManager()

        assert pm.get_resume_point() is None

        pm.mark_processed("item_001")
        pm.mark_processed("item_002")

        assert pm.get_resume_point() == "item_002"

    def test_save_locally(self, tmp_path):
        """Test saving progress locally."""
        pm = ProgressManager()
        pm.local_path = str(tmp_path)

        pm.mark_processed("item_001")
        pm.save(upload_to_hf=False)

        progress_file = tmp_path / "synthesis_progress.json"
        assert progress_file.exists()

        with open(progress_file) as f:
            saved = json.load(f)
        assert saved['total_processed'] == 1

    def test_convert_types(self):
        """Test type conversion for JSON serialization."""
        pm = ProgressManager()

        # Test with numpy-like types (mock)
        class MockNumpy:
            def item(self):
                return 42

        result = pm._convert_types({'value': MockNumpy()})
        assert result['value'] == 42

    def test_duplicate_processing(self):
        """Test that duplicate processing doesn't double count."""
        pm = ProgressManager()

        pm.mark_processed("item_001")
        pm.mark_processed("item_001")  # Duplicate

        assert pm.progress_data['total_processed'] == 1
