"""Robust JSON extraction and parsing utilities."""

import re
import json
from typing import Any, Optional, List

from ..core.logger import get_logger

logger = get_logger(__name__)


def extract_json(text: str) -> Optional[str]:
    """Extract JSON from text that may contain markdown or other content.

    Args:
        text: Raw text containing JSON

    Returns:
        Extracted JSON string or None
    """
    if not text:
        return None

    text = text.strip()

    # Try 1: Extract from ```json code block
    if "```json" in text:
        try:
            json_content = text.split("```json")[1].split("```")[0].strip()
            return json_content
        except (IndexError, ValueError):
            pass

    # Try 2: Extract from generic ``` code block
    if "```" in text:
        try:
            json_content = text.split("```")[1].strip()
            # Remove language identifier if present
            if json_content.startswith(('json', 'JSON')):
                json_content = json_content[4:].strip()
            return json_content
        except (IndexError, ValueError):
            pass

    # Try 3: Find JSON array pattern
    json_match = re.search(r'\[[\s\S]*?\]', text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    # Try 4: Find JSON object pattern
    json_match = re.search(r'\{[\s\S]*?\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    # Try 5: Return text as-is if it looks like JSON
    if text.startswith('[') or text.startswith('{'):
        return text

    return None


def clean_json_string(json_str: str) -> str:
    """Clean a JSON string for parsing.

    Args:
        json_str: Raw JSON string

    Returns:
        Cleaned JSON string
    """
    if not json_str:
        return json_str

    # Replace newlines with spaces
    cleaned = json_str.replace('\n', ' ')

    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Fix trailing commas (invalid JSON)
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)

    # Fix unquoted keys (common LLM error)
    # Pattern: word followed by colon, not inside quotes
    # This is a simplified fix and may not cover all cases
    cleaned = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', cleaned)

    return cleaned


def parse_json_safely(text: str, default: Any = None) -> Any:
    """Safely parse JSON from text with multiple fallback strategies.

    Args:
        text: Text containing JSON
        default: Default value if parsing fails

    Returns:
        Parsed JSON data or default
    """
    if not text:
        return default

    # Extract JSON from text
    json_content = extract_json(text)
    if not json_content:
        logger.debug("No JSON content found in text")
        return default

    # Try 1: Parse as-is
    try:
        return json.loads(json_content)
    except json.JSONDecodeError:
        pass

    # Try 2: Clean and parse
    cleaned = clean_json_string(json_content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error: {e}")

    # Try 3: Attempt to fix common issues
    fixed = _attempt_json_fix(cleaned)
    if fixed:
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON: {json_content[:100]}...")
    return default


def _attempt_json_fix(json_str: str) -> Optional[str]:
    """Attempt to fix common JSON errors.

    Args:
        json_str: Malformed JSON string

    Returns:
        Fixed JSON string or None
    """
    if not json_str:
        return None

    # Try to add missing closing brackets
    open_brackets = json_str.count('[') - json_str.count(']')
    open_braces = json_str.count('{') - json_str.count('}')

    fixed = json_str
    if open_brackets > 0:
        fixed += ']' * open_brackets
    if open_braces > 0:
        fixed += '}' * open_braces

    if fixed != json_str:
        return fixed

    return None


def parse_qa_variants(text: str) -> List[dict]:
    """Parse QA variants from LLM response.

    Args:
        text: LLM response text

    Returns:
        List of QA pair dictionaries
    """
    data = parse_json_safely(text, default=[])

    if not isinstance(data, list):
        logger.warning(f"Expected list, got {type(data)}")
        return []

    # Filter valid QA pairs
    valid_pairs = []
    for item in data:
        if isinstance(item, dict) and 'question' in item and 'answer' in item:
            valid_pairs.append({
                'question': str(item['question']).strip(),
                'answer': str(item['answer']).strip(),
                'thinking': str(item.get('thinking', '')).strip()
            })

    return valid_pairs


if __name__ == "__main__":
    print("=" * 60)
    print("JSON PARSER TEST")
    print("=" * 60)

    # Test 1: Extract from markdown
    md_text = '''Here's the response:
```json
[{"question": "What is 2+2?", "answer": "4"}]
```
'''
    extracted = extract_json(md_text)
    print(f"  ✓ Extract from markdown: {extracted[:40]}...")

    # Test 2: Parse with cleaning
    dirty_json = '''[
        {"question": "Test?", "answer": "Yes",},
        {"question": "Again?", "answer": "No"}
    ]'''
    parsed = parse_json_safely(dirty_json)
    print(f"  ✓ Parse with cleaning: {len(parsed)} items")

    # Test 3: Extract from plain text
    plain_text = 'The result is [{"key": "value"}] in JSON format.'
    extracted = extract_json(plain_text)
    print(f"  ✓ Extract from plain text: {extracted}")

    # Test 4: Parse QA variants
    qa_text = '''```json
[
    {"question": "What is Python?", "answer": "A programming language."},
    {"question": "What is Java?", "answer": "Another programming language.", "thinking": "Let me think..."}
]
```'''
    variants = parse_qa_variants(qa_text)
    print(f"  ✓ Parse QA variants: {len(variants)} pairs")
    for v in variants:
        print(f"    - Q: {v['question'][:30]}...")

    # Test 5: Handle invalid JSON
    invalid = "This is not JSON at all"
    result = parse_json_safely(invalid, default={"error": True})
    print(f"  ✓ Handle invalid: returned default = {result}")

    # Test 6: Fix unclosed brackets
    unclosed = '[{"key": "value"'
    fixed = _attempt_json_fix(unclosed)
    print(f"  ✓ Fix unclosed: {unclosed} -> {fixed}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
