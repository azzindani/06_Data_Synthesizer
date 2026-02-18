"""Dynamic DataFrame construction from schema configuration.

Each synthesis type (QA, DeepThinking, Corpus, SourceQA, …) produces rows
with different columns and different measurements.  DynamicSchemaBuilder
reads the ``schema`` section of a domain config YAML and constructs a
consistently typed, correctly ordered pandas DataFrame from raw result dicts.

If no schema is defined (schema.is_empty), the builder falls back to a
passthrough that includes every key from the result dicts — useful during
development or when all columns are desired.
"""

import pandas as pd
from typing import Any, Dict, List, Optional

from .config import SchemaConfig, SchemaColumnDef


# ---------------------------------------------------------------------------
# Type casting helpers
# ---------------------------------------------------------------------------

_CAST_MAP = {
    "string": str,
    "str": str,
    "int": int,
    "integer": int,
    "float": float,
    "double": float,
    "bool": bool,
    "boolean": bool,
}


def _cast(value: Any, type_str: str) -> Any:
    """Cast *value* to the type described by *type_str*.

    Returns *value* unchanged if the cast fails or the type is unknown.
    """
    if value is None:
        return None
    cast_fn = _CAST_MAP.get(type_str.lower())
    if cast_fn is None:
        return value
    try:
        return cast_fn(value)
    except (ValueError, TypeError):
        return value


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class DynamicSchemaBuilder:
    """Build pandas DataFrames from List[dict] result rows using a SchemaConfig.

    Usage::

        builder = DynamicSchemaBuilder(config.schema)
        df = builder.build_dataframe(results)
        df.to_parquet("out.parquet", index=False)
    """

    def __init__(self, schema: SchemaConfig):
        self.schema = schema

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Construct a DataFrame from *results* according to the schema.

        Behaviour:
        - If the schema has no columns defined, all keys in the result dicts
          are used as-is (passthrough / development mode).
        - If the schema IS defined, only the configured columns are included,
          in the configured order, with the configured types.
        - Optional columns absent from a result row receive their default value.
        - Required columns absent from a result row are filled with ``None``
          (no exception is raised; the caller can validate afterward).

        Args:
            results: List of result dicts, one per generated sample.

        Returns:
            A pandas DataFrame ready to persist as Parquet.
        """
        if not results:
            return self._empty_dataframe()

        if self.schema.is_empty:
            return pd.DataFrame(results)

        rows = [self._build_row(r) for r in results]
        df = pd.DataFrame(rows)

        # Enforce column order from schema
        ordered_cols = [c.name for c in self.schema.columns if c.name in df.columns]
        return df[ordered_cols]

    def column_names(self) -> List[str]:
        """Return the list of output column names defined in the schema."""
        return self.schema.column_names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for col_def in self.schema.columns:
            raw = result.get(col_def.name)
            if raw is None:
                raw = col_def.default
            row[col_def.name] = _cast(raw, col_def.type)
        return row

    def _empty_dataframe(self) -> pd.DataFrame:
        if self.schema.is_empty:
            return pd.DataFrame()
        return pd.DataFrame(columns=self.schema.column_names)

    # ------------------------------------------------------------------
    # Convenience: validate a result list against the schema
    # ------------------------------------------------------------------

    def validate_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Return a list of warning strings for missing required columns.

        Does NOT raise — callers decide how to handle warnings.
        """
        warnings: List[str] = []
        required = {c.name for c in self.schema.columns if c.required}
        for i, result in enumerate(results):
            for col_name in required:
                if col_name not in result or result[col_name] is None:
                    warnings.append(
                        f"Row {i}: required column '{col_name}' is missing or None"
                    )
        return warnings


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .config import SchemaColumnDef, SchemaConfig

    print("=" * 60)
    print("SCHEMA BUILDER TEST")
    print("=" * 60)

    # Build a test schema matching QA synthesis output
    schema = SchemaConfig(columns=[
        SchemaColumnDef(name="question",        type="string",  source="generated", required=True),
        SchemaColumnDef(name="answer",          type="string",  source="generated", required=True),
        SchemaColumnDef(name="thinking",        type="string",  source="generated", required=False, default=""),
        SchemaColumnDef(name="topic",           type="string",  source="metadata",  required=True),
        SchemaColumnDef(name="question_length", type="int",     source="computed",  required=False),
        SchemaColumnDef(name="answer_length",   type="int",     source="computed",  required=False),
        SchemaColumnDef(name="timestamp",       type="string",  source="metadata",  required=False),
        SchemaColumnDef(name="provider",        type="string",  source="metadata",  required=False),
    ])

    builder = DynamicSchemaBuilder(schema)

    # Simulate result rows
    sample_results = [
        {
            "question": "What is the legal age of majority in Indonesia?",
            "answer": "In Indonesia the legal age of majority is 18 years...",
            "topic": "Indonesian Legal System",
            "question_length": 48,
            "answer_length": 52,
            "timestamp": "2026-02-18T12:00:00",
            "provider": "gemini",
            # 'thinking' intentionally absent → should use default ""
        },
        {
            "question": "Explain contract law principles.",
            "answer": "Contract law in Indonesia is governed by...",
            "thinking": "Let me think step by step...",
            "topic": "Contract Law",
            "question_length": 32,
            "answer_length": 45,
            "timestamp": "2026-02-18T12:01:00",
            "provider": "gemini",
        },
    ]

    df = builder.build_dataframe(sample_results)
    print(f"  ✓ DataFrame shape: {df.shape}")
    print(f"  ✓ Columns: {list(df.columns)}")
    print(f"  ✓ Thinking default filled: '{df['thinking'].iloc[0]}'")
    assert df["thinking"].iloc[0] == ""
    assert df["thinking"].iloc[1] == "Let me think step by step..."
    assert list(df.columns) == [c.name for c in schema.columns]
    print("  ✓ Column order matches schema")

    # Empty results
    empty_df = builder.build_dataframe([])
    assert list(empty_df.columns) == schema.column_names
    print("  ✓ Empty results returns empty DataFrame with correct columns")

    # Passthrough (no schema)
    passthrough_builder = DynamicSchemaBuilder(SchemaConfig())
    pt_df = passthrough_builder.build_dataframe(sample_results)
    assert "question" in pt_df.columns and "answer" in pt_df.columns
    print("  ✓ Passthrough (no schema) works")

    # Validation
    warnings = builder.validate_results([{"question": "Q", "topic": "T"}])
    assert any("answer" in w for w in warnings), f"Expected answer warning, got {warnings}"
    print(f"  ✓ Validation warnings: {warnings}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
