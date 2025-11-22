"""Configuration loader and validation."""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    model: str
    temperature: float = 0.7
    max_output_tokens: int = 8000
    rate_limit_delay: float = 4.0
    api_key: str = ""

    def __post_init__(self):
        # Load API key from environment if not provided
        if not self.api_key:
            env_key = f"{self.name.upper()}_API_KEY"
            self.api_key = os.getenv(env_key, "")


@dataclass
class SynthesisConfig:
    """Configuration for synthesis settings."""
    type: str = "qa"  # qa | deep_thinking | corpus
    domain_config: str = ""
    batch_size: int = 5
    questions_per_topic: int = 50
    num_variants: int = 5


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
    type: str = "huggingface"  # huggingface | local | both
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

    # Domain-specific settings
    domain: Dict[str, Any] = field(default_factory=dict)

    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider."""
        return self.providers.get(name)

    def get_active_providers(self) -> List[str]:
        """Get list of active providers in priority order."""
        providers = [self.primary_provider]
        providers.extend(self.fallback_providers)
        return [p for p in providers if p in self.providers]

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Check primary provider exists
        if self.primary_provider not in self.providers:
            errors.append(f"Primary provider '{self.primary_provider}' not configured")

        # Check at least one provider has API key
        has_key = any(p.api_key for p in self.providers.values())
        if not has_key:
            errors.append("No provider has API key configured")

        # Check output repository for HuggingFace
        if self.output.type in ["huggingface", "both"]:
            if not self.output.repository:
                errors.append("HuggingFace repository not specified")
            if not self.output.huggingface_token:
                errors.append("HuggingFace token not configured")

        # Check synthesis type
        valid_types = ["qa", "deep_thinking", "corpus"]
        if self.synthesis.type not in valid_types:
            errors.append(f"Invalid synthesis type: {self.synthesis.type}")

        return errors


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        # Return default config if file doesn't exist
        return _create_default_config()

    with open(config_file, 'r') as f:
        data = yaml.safe_load(f) or {}

    return _parse_config(data)


def _create_default_config() -> Config:
    """Create default configuration."""
    config = Config()

    # Add default Gemini provider
    config.providers["gemini"] = ProviderConfig(
        name="gemini",
        model="gemini-2.5-flash",
        temperature=0.7,
        max_output_tokens=8000,
        rate_limit_delay=4.0
    )

    # Add default OpenRouter provider
    config.providers["openrouter"] = ProviderConfig(
        name="openrouter",
        model="google/gemini-flash-1.5",
        temperature=0.7,
        max_output_tokens=8000,
        rate_limit_delay=2.0
    )

    return config


def _parse_config(data: dict) -> Config:
    """Parse configuration from dictionary."""
    config = Config()

    # App settings
    app_data = data.get("app", {})
    config.app_name = app_data.get("name", config.app_name)
    config.mode = app_data.get("mode", config.mode)
    config.log_level = app_data.get("log_level", config.log_level)

    # Provider settings
    providers_data = data.get("providers", {})
    config.primary_provider = providers_data.get("primary", config.primary_provider)
    config.fallback_providers = providers_data.get("fallback", config.fallback_providers)
    config.auto_switch = providers_data.get("auto_switch", config.auto_switch)

    # Parse individual providers
    for provider_name in ["gemini", "openrouter", "openai"]:
        if provider_name in providers_data:
            provider_data = providers_data[provider_name]
            config.providers[provider_name] = ProviderConfig(
                name=provider_name,
                model=provider_data.get("model", ""),
                temperature=provider_data.get("temperature", 0.7),
                max_output_tokens=provider_data.get("max_output_tokens", 8000),
                rate_limit_delay=provider_data.get("rate_limit_delay", 4.0),
                api_key=provider_data.get("api_key", "")
            )

    # If no providers configured, add defaults
    if not config.providers:
        config = _create_default_config()
        # Re-apply other settings
        config.app_name = app_data.get("name", config.app_name)
        config.mode = app_data.get("mode", config.mode)
        config.log_level = app_data.get("log_level", config.log_level)

    # Synthesis settings
    synthesis_data = data.get("synthesis", {})
    config.synthesis = SynthesisConfig(
        type=synthesis_data.get("type", "qa"),
        domain_config=synthesis_data.get("domain_config", ""),
        batch_size=synthesis_data.get("batch_size", 5),
        questions_per_topic=synthesis_data.get("questions_per_topic", 50),
        num_variants=synthesis_data.get("num_variants", 5)
    )

    # Quality settings
    quality_data = data.get("quality", {})
    config.quality = QualityConfig(
        min_answer_length=quality_data.get("min_answer_length", 100),
        max_answer_length=quality_data.get("max_answer_length", 2000),
        min_question_length=quality_data.get("min_question_length", 15),
        max_question_length=quality_data.get("max_question_length", 1500),
        target_language=quality_data.get("target_language", "id"),
        language_confidence=quality_data.get("language_confidence", 0.8),
        min_quality_score=quality_data.get("min_quality_score", 0.6)
    )

    # Retry settings
    retry_data = data.get("retry", {})
    config.retry = RetryConfig(
        max_attempts=retry_data.get("max_attempts", 5),
        base_delay=retry_data.get("base_delay", 2.0),
        max_delay=retry_data.get("max_delay", 60.0),
        exponential_base=retry_data.get("exponential_base", 2.0)
    )

    # Output settings
    output_data = data.get("output", {})
    config.output = OutputConfig(
        type=output_data.get("type", "huggingface"),
        repository=output_data.get("repository", ""),
        local_path=output_data.get("local_path", "./output"),
        chunk_format=output_data.get("chunk_format", "parquet"),
        progress_file=output_data.get("progress_file", "synthesis_progress.json"),
        huggingface_token=output_data.get("huggingface_token", "")
    )

    # Domain settings (pass through)
    config.domain = data.get("domain", {})

    return config


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG MODULE TEST")
    print("=" * 60)

    # Test 1: Default config creation
    config = _create_default_config()
    print(f"  ✓ Default config created")
    print(f"    - Primary provider: {config.primary_provider}")
    print(f"    - Providers: {list(config.providers.keys())}")

    # Test 2: Validation
    errors = config.validate()
    print(f"  ✓ Validation completed: {len(errors)} errors")
    for err in errors:
        print(f"    - {err}")

    # Test 3: Get active providers
    active = config.get_active_providers()
    print(f"  ✓ Active providers: {active}")

    # Test 4: Provider config retrieval
    gemini = config.get_provider_config("gemini")
    if gemini:
        print(f"  ✓ Gemini config: model={gemini.model}, temp={gemini.temperature}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
