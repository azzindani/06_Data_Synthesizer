"""Configuration loader and validation.

Supports ${VAR} and ${VAR:-default} environment variable interpolation in all
string values across both the main config.yaml and domain config YAMLs.
"""

import os
import re
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------

def interpolate_env_vars(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} placeholders using os.environ.

    Supported syntax:
        ${VAR}            → os.environ["VAR"]  (empty string if not set)
        ${VAR:-fallback}  → os.environ["VAR"]  or "fallback" if not set
    """
    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(3)  # None when no :- present
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        return ""  # undefined with no default → empty string

    return re.sub(r"\$\{([^}:]+?)(?::-(.*?))?\}", _replace, value)


def apply_env_interpolation(data: Any) -> Any:
    """Recursively apply env var interpolation to every string in *data*."""
    if isinstance(data, str):
        return interpolate_env_vars(data)
    if isinstance(data, dict):
        return {k: apply_env_interpolation(v) for k, v in data.items()}
    if isinstance(data, list):
        return [apply_env_interpolation(v) for v in data]
    return data


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    model: str
    temperature: float = 0.7
    max_output_tokens: int = 65536
    top_p: float = 0.8
    top_k: int = 40
    rate_limit_delay: float = 4.0
    api_key: str = ""

    def __post_init__(self):
        # Load API key from environment if not provided directly.
        # Accept either ``{NAME}_API_KEY`` (single / comma-separated) or
        # ``{NAME}_API_KEYS`` (plural). Either form satisfies the validator;
        # providers that support multi-key rotation (Gemini, OpenRouter)
        # split commas + read the plural variant themselves.
        if not self.api_key:
            name_upper = self.name.upper()
            self.api_key = (
                os.getenv(f"{name_upper}_API_KEY")
                or os.getenv(f"{name_upper}_API_KEYS")
                or ""
            )


@dataclass
class MetaConfig:
    """Metadata describing what a config file is for."""
    name: str = ""
    description: str = ""
    version: str = "1.0"
    # qa | deep_thinking | corpus | source_qa | source_corpus
    synthesis_type: str = "qa"


@dataclass
class RunConfig:
    """All per-run parameters that control a synthesis job.

    These are typically set in the domain config YAML (run_config section)
    and override the global defaults from config.yaml when merged.
    """
    # ---- Model / API ----
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 65536
    top_p: float = 0.8
    top_k: int = 40
    rate_limit_delay: float = 4.0

    # ---- HuggingFace I/O ----
    input_repository: str = ""     # source dataset repo (for source-based runs)
    output_repository: str = ""    # destination dataset repo

    # ---- Column mappings ----
    # Output column names (what appears in the saved DataFrame / parquet)
    question_column: str = "question"
    answer_column: str = "answer"
    thinking_column: str = "thinking"
    instruction_column: str = ""      # leave empty if not used
    corpus_column: str = "text"       # source column in *input* dataset
    id_column: str = "id"

    # ---- Processing ----
    batch_size: int = 5
    questions_per_topic: int = 50
    chunk_size: int = 20
    skip_processed_rows: bool = True
    progress_file: str = "synthesis_progress.json"
    target_language_confidence: float = 0.8

    # ---- Safety ----
    safety_settings: Dict[str, str] = field(default_factory=lambda: {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
    })


@dataclass
class SchemaColumnDef:
    """Definition of a single column in the output DataFrame."""
    name: str                        # output column name
    type: str = "string"             # string | int | float | bool
    # generated | input | metadata | computed
    source: str = "generated"
    required: bool = True
    default: Any = None              # value when column is absent or None


@dataclass
class SchemaConfig:
    """Defines which columns appear in the output DataFrame and their types."""
    columns: List[SchemaColumnDef] = field(default_factory=list)

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    @property
    def is_empty(self) -> bool:
        return len(self.columns) == 0


@dataclass
class SynthesisConfig:
    """Global synthesis settings."""
    type: str = "qa"           # qa | deep_thinking | corpus | source_qa | source_corpus
    domain_config: str = ""    # path to domain config YAML
    batch_size: int = 5
    questions_per_topic: int = 50
    num_variants: int = 5
    # Hard ceiling on total generated rows. When ``total_generated >= target_rows``
    # the synthesis loop stops cleanly (progress saved, container exits 0).
    # Leave ``None`` (or 0) to use the implicit ``topics × questions_per_topic`` limit.
    target_rows: Optional[int] = None


@dataclass
class QualityConfig:
    """Configuration for quality validation."""
    min_answer_length: int = 100
    max_answer_length: int = 2000
    min_question_length: int = 15
    max_question_length: int = 1500
    target_language: str = "id"
    language_confidence: float = 0.8
    min_quality_score: float = 0.6


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


@dataclass
class OutputConfig:
    """Configuration for output handling."""
    type: str = "huggingface"      # huggingface | local | both
    repository: str = ""
    local_path: str = "./output"
    chunk_format: str = "parquet"
    progress_file: str = "synthesis_progress.json"
    huggingface_token: str = ""

    def __post_init__(self):
        if not self.huggingface_token:
            self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN", "")


@dataclass
class Config:
    """Main configuration container."""
    app_name: str = "data-synthesizer"
    mode: str = "production"
    log_level: str = "INFO"
    log_file: str = "logs/synthesizer.log"

    # Provider settings
    primary_provider: str = "gemini"
    fallback_providers: List[str] = field(default_factory=lambda: ["openrouter"])
    auto_switch: bool = True
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)

    # Sub-configs
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # New top-level configs (populated from domain config YAML)
    meta: MetaConfig = field(default_factory=MetaConfig)
    run_config: RunConfig = field(default_factory=RunConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)

    # Pass-through dicts from domain config YAML
    domain: Dict[str, Any] = field(default_factory=dict)
    prompts: Dict[str, Any] = field(default_factory=dict)
    thinking_requirements: Dict[str, Any] = field(default_factory=dict)

    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        return self.providers.get(name)

    def get_active_providers(self) -> List[str]:
        providers = [self.primary_provider]
        providers.extend(self.fallback_providers)
        return [p for p in providers if p in self.providers]

    def validate(self) -> List[str]:
        errors = []

        if self.primary_provider not in self.providers:
            errors.append(f"Primary provider '{self.primary_provider}' not configured")

        has_key = any(p.api_key for p in self.providers.values())
        if not has_key:
            errors.append("No provider has API key configured")

        if self.output.type in ["huggingface", "both"]:
            if not self.output.repository:
                errors.append("HuggingFace output repository not specified")
            if not self.output.huggingface_token:
                errors.append("HuggingFace token not configured")

        valid_types = [
            "qa", "deep_thinking", "corpus",
            "source_qa", "source_corpus",
        ]
        effective_type = self.meta.synthesis_type or self.synthesis.type
        if effective_type not in valid_types:
            errors.append(f"Invalid synthesis type: {effective_type}")

        return errors


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml",
                domain_config_path: str = "") -> Config:
    """Load configuration from YAML file(s).

    Args:
        config_path: Path to the main config.yaml.
        domain_config_path: Optional path to a domain config YAML.
            Overrides ``synthesis.domain_config`` from the main config.

    Returns:
        Populated Config object with env vars already interpolated.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        config = _create_default_config()
    else:
        with open(config_file, "r") as f:
            raw = yaml.safe_load(f) or {}
        data = apply_env_interpolation(raw)
        config = _parse_config(data)

    # Determine domain config path
    domain_path = domain_config_path or config.synthesis.domain_config
    if domain_path:
        domain_file = Path(domain_path)
        if domain_file.exists():
            with open(domain_file, "r") as f:
                raw_domain = yaml.safe_load(f) or {}
            domain_data = apply_env_interpolation(raw_domain)
            _merge_domain_config(config, domain_data)

    return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_default_config() -> Config:
    config = Config()
    config.providers["gemini"] = ProviderConfig(
        name="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        max_output_tokens=65536,
        top_p=0.8,
        top_k=40,
        rate_limit_delay=4.0,
    )
    config.providers["openrouter"] = ProviderConfig(
        name="openrouter",
        model="google/gemini-flash-1.5",
        temperature=0.7,
        max_output_tokens=65536,
        top_p=0.8,
        top_k=40,
        rate_limit_delay=2.0,
    )
    return config


def _parse_config(data: dict) -> Config:
    config = Config()

    app_data = data.get("app", {})
    config.app_name = app_data.get("name", config.app_name)
    config.mode = app_data.get("mode", config.mode)
    config.log_level = app_data.get("log_level", config.log_level)

    providers_data = data.get("providers", {})
    config.primary_provider = providers_data.get("primary", config.primary_provider)
    config.fallback_providers = providers_data.get("fallback", config.fallback_providers)
    config.auto_switch = providers_data.get("auto_switch", config.auto_switch)

    for provider_name, provider_data in providers_data.items():
        if provider_name in {"primary", "fallback", "auto_switch"}:
            continue
        if not isinstance(provider_data, dict):
            continue
        config.providers[provider_name] = ProviderConfig(
            name=provider_name,
            model=provider_data.get("model", ""),
            temperature=provider_data.get("temperature", 0.7),
            max_output_tokens=provider_data.get("max_output_tokens", 65536),
            top_p=provider_data.get("top_p", 0.8),
            top_k=provider_data.get("top_k", 40),
            rate_limit_delay=provider_data.get("rate_limit_delay", 4.0),
            api_key=provider_data.get("api_key", ""),
        )

    if not config.providers:
        config = _create_default_config()
        config.app_name = app_data.get("name", config.app_name)
        config.mode = app_data.get("mode", config.mode)
        config.log_level = app_data.get("log_level", config.log_level)

    synthesis_data = data.get("synthesis", {})
    config.synthesis = SynthesisConfig(
        type=synthesis_data.get("type", "qa"),
        domain_config=synthesis_data.get("domain_config", ""),
        batch_size=synthesis_data.get("batch_size", 5),
        questions_per_topic=synthesis_data.get("questions_per_topic", 50),
        num_variants=synthesis_data.get("num_variants", 5),
    )

    quality_data = data.get("quality", {})
    config.quality = QualityConfig(
        min_answer_length=quality_data.get("min_answer_length", 100),
        max_answer_length=quality_data.get("max_answer_length", 2000),
        min_question_length=quality_data.get("min_question_length", 15),
        max_question_length=quality_data.get("max_question_length", 1500),
        target_language=quality_data.get("target_language", "id"),
        language_confidence=quality_data.get("language_confidence", 0.8),
        min_quality_score=quality_data.get("min_quality_score", 0.6),
    )

    retry_data = data.get("retry", {})
    config.retry = RetryConfig(
        max_attempts=retry_data.get("max_attempts", 5),
        base_delay=retry_data.get("base_delay", 2.0),
        max_delay=retry_data.get("max_delay", 60.0),
        exponential_base=retry_data.get("exponential_base", 2.0),
    )

    output_data = data.get("output", {})
    config.output = OutputConfig(
        type=output_data.get("type", "huggingface"),
        repository=output_data.get("repository", ""),
        local_path=output_data.get("local_path", "./output"),
        chunk_format=output_data.get("chunk_format", "parquet"),
        progress_file=output_data.get("progress_file", "synthesis_progress.json"),
        huggingface_token=output_data.get("huggingface_token", ""),
    )

    config.domain = data.get("domain", {})
    return config


def _parse_run_config(run_data: dict) -> RunConfig:
    """Build a RunConfig from a raw dict (run_config or notebook_config section)."""
    safety_defaults = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
    }
    return RunConfig(
        model_name=run_data.get("model_name", "gemini-2.5-flash"),
        temperature=run_data.get("temperature", 0.7),
        max_output_tokens=run_data.get("max_output_tokens", 65536),
        top_p=run_data.get("top_p", 0.8),
        top_k=run_data.get("top_k", 40),
        rate_limit_delay=run_data.get("rate_limit_delay", 4.0),
        input_repository=run_data.get("input_repository", ""),
        output_repository=run_data.get("output_repository", ""),
        question_column=run_data.get("question_column", "question"),
        answer_column=run_data.get("answer_column", "answer"),
        thinking_column=run_data.get("thinking_column", "thinking"),
        instruction_column=run_data.get("instruction_column", ""),
        corpus_column=run_data.get("corpus_column", "text"),
        id_column=run_data.get("id_column", "id"),
        batch_size=run_data.get("batch_size", 5),
        questions_per_topic=run_data.get("questions_per_topic", 50),
        chunk_size=run_data.get("chunk_size", 20),
        skip_processed_rows=run_data.get("skip_processed_rows", True),
        progress_file=run_data.get("progress_file", "synthesis_progress.json"),
        target_language_confidence=run_data.get("target_language_confidence", 0.8),
        safety_settings=run_data.get("safety_settings", safety_defaults),
    )


def _parse_schema(schema_data: dict) -> SchemaConfig:
    """Build a SchemaConfig from the schema section of a domain config."""
    columns = []
    for col in schema_data.get("columns", []):
        if not isinstance(col, dict) or "name" not in col:
            continue
        columns.append(SchemaColumnDef(
            name=col["name"],
            type=col.get("type", "string"),
            source=col.get("source", "generated"),
            required=col.get("required", True),
            default=col.get("default", None),
        ))
    return SchemaConfig(columns=columns)


def _merge_domain_config(config: Config, domain_data: dict) -> None:
    """Merge a loaded domain config YAML into *config* in-place.

    Priority order (highest → lowest):
        domain YAML  >  main config.yaml  >  dataclass defaults
    """
    # --- meta ---
    meta_data = domain_data.get("meta", {})
    if meta_data:
        config.meta = MetaConfig(
            name=meta_data.get("name", ""),
            description=meta_data.get("description", ""),
            version=meta_data.get("version", "1.0"),
            synthesis_type=meta_data.get("synthesis_type", "qa"),
        )
        # Sync synthesis type
        config.synthesis.type = config.meta.synthesis_type

    # --- run_config (falls back to notebook_config for backward compat) ---
    run_data = domain_data.get("run_config") or domain_data.get("notebook_config") or {}
    if run_data:
        config.run_config = _parse_run_config(run_data)

        # Propagate model settings to the active provider config
        primary = config.providers.get(config.primary_provider)
        if primary:
            if "model_name" in run_data:
                primary.model = run_data["model_name"]
            if "temperature" in run_data:
                primary.temperature = run_data["temperature"]
            if "max_output_tokens" in run_data:
                primary.max_output_tokens = run_data["max_output_tokens"]
            if "top_p" in run_data:
                primary.top_p = run_data["top_p"]
            if "top_k" in run_data:
                primary.top_k = run_data["top_k"]
            if "rate_limit_delay" in run_data:
                primary.rate_limit_delay = run_data["rate_limit_delay"]

        # Propagate output settings
        if run_data.get("output_repository"):
            config.output.repository = run_data["output_repository"]
        if run_data.get("progress_file"):
            config.output.progress_file = run_data["progress_file"]

        # Propagate synthesis settings
        if "batch_size" in run_data:
            config.synthesis.batch_size = run_data["batch_size"]
        if "questions_per_topic" in run_data:
            config.synthesis.questions_per_topic = run_data["questions_per_topic"]

    # --- schema ---
    schema_data = domain_data.get("schema", {})
    if schema_data:
        config.schema = _parse_schema(schema_data)

    # --- quality overrides ---
    quality_data = domain_data.get("quality", {})
    for key, value in quality_data.items():
        attr = key.replace("-", "_")
        if hasattr(config.quality, attr):
            setattr(config.quality, attr, value)

    # --- pass-through sections ---
    if domain_data.get("domain"):
        config.domain = domain_data["domain"]
    if domain_data.get("prompts"):
        config.prompts = domain_data["prompts"]
    if domain_data.get("thinking_requirements"):
        config.thinking_requirements = domain_data["thinking_requirements"]


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG MODULE TEST")
    print("=" * 60)

    config = _create_default_config()
    print(f"  ✓ Default config created")
    print(f"    - Primary provider: {config.primary_provider}")
    print(f"    - Providers: {list(config.providers.keys())}")
    print(f"    - run_config.question_column: {config.run_config.question_column}")
    print(f"    - schema columns: {config.schema.column_names}")

    errors = config.validate()
    print(f"  ✓ Validation: {len(errors)} errors")

    active = config.get_active_providers()
    print(f"  ✓ Active providers: {active}")

    gemini = config.get_provider_config("gemini")
    if gemini:
        print(f"  ✓ Gemini: model={gemini.model}, temp={gemini.temperature}, "
              f"top_p={gemini.top_p}, top_k={gemini.top_k}, "
              f"max_tokens={gemini.max_output_tokens}")

    # Test env var interpolation
    os.environ["TEST_REPO"] = "user/my-dataset"
    result = interpolate_env_vars("${TEST_REPO:-default/repo}")
    assert result == "user/my-dataset", f"Expected user/my-dataset, got {result}"
    result2 = interpolate_env_vars("${UNDEFINED_VAR:-fallback}")
    assert result2 == "fallback", f"Expected fallback, got {result2}"
    result3 = interpolate_env_vars("${UNDEFINED_NO_DEFAULT}")
    assert result3 == "", f"Expected empty string, got {result3}"
    print("  ✓ Env var interpolation works correctly")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
