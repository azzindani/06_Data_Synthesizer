"""Unit tests for JSON parser utilities."""

import pytest

from src.utils.json_parser import (
    extract_json, clean_json_string, parse_json_safely,
    parse_qa_variants, _attempt_json_fix
)


class TestExtractJson:
    """Tests for extract_json function."""

    def test_extract_from_markdown_json(self):
        """Test extracting JSON from markdown code block."""
        text = '''Here is the result:
```json
[{"key": "value"}]
```
'''
        result = extract_json(text)
        assert result == '[{"key": "value"}]'

    def test_extract_from_generic_code_block(self):
        """Test extracting from generic code block."""
        text = '''```
{"key": "value"}
```'''
        result = extract_json(text)
        assert '{"key": "value"}' in result

    def test_extract_array_from_text(self):
        """Test extracting array from plain text."""
        text = 'The result is [1, 2, 3] and more text.'
        result = extract_json(text)
        assert result == '[1, 2, 3]'

    def test_extract_object_from_text(self):
        """Test extracting object from plain text."""
        text = 'Here is {"a": 1} the object.'
        result = extract_json(text)
        assert result == '{"a": 1}'

    def test_return_none_for_no_json(self):
        """Test returning None when no JSON found."""
        text = 'This is plain text without any JSON.'
        result = extract_json(text)
        assert result is None

    def test_empty_text(self):
        """Test handling empty text."""
        assert extract_json("") is None
        assert extract_json(None) is None


class TestCleanJsonString:
    """Tests for clean_json_string function."""

    def test_remove_newlines(self):
        """Test removing newlines."""
        json_str = '{\n"key":\n"value"\n}'
        result = clean_json_string(json_str)
        assert '\n' not in result

    def test_fix_trailing_comma_object(self):
        """Test fixing trailing comma in object."""
        json_str = '{"key": "value",}'
        result = clean_json_string(json_str)
        assert ',}' not in result

    def test_fix_trailing_comma_array(self):
        """Test fixing trailing comma in array."""
        json_str = '[1, 2, 3,]'
        result = clean_json_string(json_str)
        assert ',]' not in result

    def test_normalize_whitespace(self):
        """Test normalizing whitespace."""
        json_str = '{"key":    "value"}'
        result = clean_json_string(json_str)
        assert '    ' not in result


class TestParseJsonSafely:
    """Tests for parse_json_safely function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        text = '[{"key": "value"}]'
        result = parse_json_safely(text)
        assert result == [{"key": "value"}]

    def test_parse_with_markdown(self):
        """Test parsing JSON wrapped in markdown."""
        text = '''```json
[{"key": "value"}]
```'''
        result = parse_json_safely(text)
        assert result == [{"key": "value"}]

    def test_return_default_on_failure(self):
        """Test returning default value on failure."""
        text = "not json"
        result = parse_json_safely(text, default={"error": True})
        assert result == {"error": True}

    def test_parse_dirty_json(self):
        """Test parsing JSON with trailing commas."""
        text = '[{"key": "value",}]'
        result = parse_json_safely(text)
        assert result == [{"key": "value"}]


class TestParseQaVariants:
    """Tests for parse_qa_variants function."""

    def test_parse_valid_variants(self, sample_qa_response):
        """Test parsing valid QA variants."""
        variants = parse_qa_variants(sample_qa_response)
        assert len(variants) == 2
        assert variants[0]['question'] == "What is Python?"
        assert 'answer' in variants[0]

    def test_parse_with_thinking(self):
        """Test parsing variants with thinking field."""
        text = '''[{"question": "Q", "answer": "A", "thinking": "Let me think..."}]'''
        variants = parse_qa_variants(text)
        assert len(variants) == 1
        assert variants[0]['thinking'] == "Let me think..."

    def test_filter_invalid_items(self):
        """Test filtering invalid items."""
        text = '''[
            {"question": "Valid", "answer": "Valid answer"},
            {"question": "Missing answer"},
            {"invalid": "object"}
        ]'''
        variants = parse_qa_variants(text)
        assert len(variants) == 1

    def test_return_empty_for_non_list(self):
        """Test returning empty list for non-list response."""
        text = '{"not": "an array"}'
        variants = parse_qa_variants(text)
        assert variants == []


class TestAttemptJsonFix:
    """Tests for _attempt_json_fix function."""

    def test_fix_unclosed_bracket(self):
        """Test fixing unclosed bracket."""
        json_str = '[{"key": "value"'
        result = _attempt_json_fix(json_str)
        assert result.endswith('}]')

    def test_fix_unclosed_brace(self):
        """Test fixing unclosed brace."""
        json_str = '{"key": "value"'
        result = _attempt_json_fix(json_str)
        assert result.endswith('}')

    def test_no_fix_needed(self):
        """Test when no fix is needed."""
        json_str = '{"key": "value"}'
        result = _attempt_json_fix(json_str)
        assert result is None
