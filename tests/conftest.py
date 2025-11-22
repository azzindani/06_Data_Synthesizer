"""Shared test fixtures and configuration."""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def default_config():
    """Create default configuration for testing."""
    from src.core.config import _create_default_config
    return _create_default_config()


@pytest.fixture
def config_with_keys(default_config):
    """Configuration with API keys from environment."""
    if 'gemini' in default_config.providers:
        default_config.providers['gemini'].api_key = os.getenv("GEMINI_API_KEY", "test_key")
    if 'openrouter' in default_config.providers:
        default_config.providers['openrouter'].api_key = os.getenv("OPENROUTER_API_KEY", "")
    return default_config


@pytest.fixture
def sample_qa_response():
    """Sample QA response from LLM."""
    return '''```json
[
    {"question": "What is Python?", "answer": "Python is a high-level programming language known for its simplicity and readability."},
    {"question": "What is JavaScript?", "answer": "JavaScript is a programming language primarily used for web development."}
]
```'''


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)
