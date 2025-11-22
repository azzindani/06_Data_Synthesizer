# Data Synthesizer

A production-ready, lightweight data synthesis pipeline designed to run continuously on low-resource VMs (GCP/AWS free tier). Supports multiple LLM providers with automatic failover.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SYNTHESIZER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Config    │    │   Logger    │    │  Progress   │         │
│  │   Loader    │    │   Manager   │    │   Tracker   │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│                    ┌───────▼───────┐                            │
│                    │  Synthesizer  │                            │
│                    │    Engine     │                            │
│                    └───────┬───────┘                            │
│                            │                                    │
│              ┌─────────────┼─────────────┐                      │
│              │             │             │                      │
│       ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐              │
│       │  QA Synth   │ │  Deep   │ │   Corpus    │              │
│       │             │ │ Thinking│ │ Generator   │              │
│       └──────┬──────┘ └────┬────┘ └──────┬──────┘              │
│              │             │             │                      │
│              └─────────────┼─────────────┘                      │
│                            │                                    │
│                    ┌───────▼───────┐                            │
│                    │   Provider    │                            │
│                    │   Factory     │                            │
│                    └───────┬───────┘                            │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                 │
│         │                  │                  │                 │
│   ┌─────▼─────┐     ┌──────▼──────┐    ┌─────▼─────┐           │
│   │  Gemini   │     │ OpenRouter  │    │  OpenAI   │           │
│   │ Provider  │     │  Provider   │    │ Provider  │           │
│   └─────┬─────┘     └──────┬──────┘    └─────┬─────┘           │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│                    ┌───────▼───────┐                            │
│                    │   Validator   │                            │
│                    │   Pipeline    │                            │
│                    └───────┬───────┘                            │
│                            │                                    │
│                    ┌───────▼───────┐                            │
│                    │    Output     │                            │
│                    │  (HF/Local)   │                            │
│                    └───────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Config → Load Domain → Select Provider → Generate Content → Validate → Save → Update Progress → Loop
           ↓                ↓                                  ↓
        Topics/         Gemini/                           Language/
        Prompts        OpenRouter                         Quality
```

---

## Features

### Core Features (MVP)
- [x] Multi-provider LLM support (Gemini, OpenRouter, OpenAI)
- [x] Automatic provider failover on errors/rate limits
- [x] Persistent progress tracking (survives restarts)
- [x] HuggingFace Hub auto-upload
- [x] Configurable synthesis modes (QA, DeepThinking, Corpus)
- [x] Language detection and validation
- [x] Rate limiting with exponential backoff

### Production Features
- [ ] Structured logging (file + console)
- [ ] Health check endpoint for monitoring
- [ ] Graceful shutdown handling (SIGTERM/SIGINT)
- [ ] Cost tracking per provider
- [ ] Dry run mode for testing
- [ ] Email/Slack notifications on completion/error

### Advanced Features
- [ ] Web dashboard for monitoring
- [ ] Multi-instance coordination
- [ ] Dynamic provider load balancing
- [ ] Custom prompt templates via config
- [ ] Resume from specific checkpoint

---

## Directory Structure

```
data_synthesizer/
├── README.md                    # This file
├── WORKFLOW.md                  # Development workflow guide
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── config.yaml                  # Main configuration file
├── .env.example                 # Environment variables template
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                    # Core business logic
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration loader & validation
│   │   ├── logger.py            # Logging setup
│   │   └── progress.py          # Progress tracking & persistence
│   │
│   ├── providers/               # LLM provider implementations
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base provider
│   │   ├── gemini.py            # Google Gemini API
│   │   ├── openrouter.py        # OpenRouter API
│   │   ├── openai.py            # OpenAI API
│   │   └── factory.py           # Provider factory with failover
│   │
│   ├── synthesizers/            # Synthesis engines
│   │   ├── __init__.py
│   │   ├── base.py              # Base synthesizer class
│   │   ├── qa_synthesis.py      # Basic QA generation
│   │   ├── deep_thinking.py     # QA with reasoning traces
│   │   └── corpus_generator.py  # Full document generation
│   │
│   ├── validators/              # Quality validation
│   │   ├── __init__.py
│   │   ├── language.py          # Language detection
│   │   ├── quality.py           # Quality scoring
│   │   └── pipeline.py          # Validation pipeline
│   │
│   ├── outputs/                 # Output handlers
│   │   ├── __init__.py
│   │   ├── huggingface.py       # HuggingFace Hub upload
│   │   ├── local.py             # Local file storage
│   │   └── base.py              # Abstract output handler
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── retry.py             # Retry with backoff
│       ├── json_parser.py       # Robust JSON extraction
│       └── text_cleaner.py      # Text preprocessing
│
├── notebooks/                   # Jupyter notebook versions
│   ├── qa_synthesis.ipynb
│   ├── deep_thinking.ipynb
│   └── corpus_generator.ipynb
│
├── configs/                     # Domain-specific configs
│   ├── indonesian_legal.yaml
│   ├── contract_generation.yaml
│   └── generic_qa.yaml
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── unit/                    # Unit tests (no external deps)
│   │   ├── test_config.py
│   │   ├── test_progress.py
│   │   ├── test_validators.py
│   │   ├── test_json_parser.py
│   │   └── test_retry.py
│   └── integration/             # Integration tests
│       ├── test_providers.py
│       ├── test_synthesizers.py
│       └── test_outputs.py
│
├── scripts/                     # Utility scripts
│   ├── run.py                   # Main entry point
│   ├── setup_vm.sh              # VM setup script
│   └── health_check.py          # Health monitoring
│
└── logs/                        # Log files (gitignored)
    └── .gitkeep
```

---

## Configuration

### Main Configuration (config.yaml)

```yaml
# General settings
app:
  name: "data-synthesizer"
  mode: "production"  # development | production
  log_level: "INFO"

# Provider configuration
providers:
  primary: "gemini"
  fallback: ["openrouter", "openai"]
  auto_switch: true

  gemini:
    model: "gemini-2.5-flash"
    temperature: 0.7
    max_output_tokens: 8000
    rate_limit_delay: 4  # seconds

  openrouter:
    model: "google/gemini-flash-1.5"
    temperature: 0.7
    max_tokens: 8000
    rate_limit_delay: 2

  openai:
    model: "gpt-4o-mini"
    temperature: 0.7
    max_tokens: 8000
    rate_limit_delay: 1

# Synthesis settings
synthesis:
  type: "qa"  # qa | deep_thinking | corpus
  domain_config: "configs/indonesian_legal.yaml"
  batch_size: 5
  questions_per_topic: 50
  num_variants: 5

# Output settings
output:
  type: "huggingface"  # huggingface | local | both
  repository: "username/dataset-name"
  local_path: "./output"
  chunk_format: "parquet"

# Quality settings
quality:
  min_answer_length: 100
  max_answer_length: 2000
  target_language: "id"
  language_confidence: 0.8
  min_quality_score: 0.6

# Retry settings
retry:
  max_attempts: 5
  base_delay: 2  # seconds
  max_delay: 60
  exponential_base: 2

# Monitoring
monitoring:
  health_check_port: 8080
  progress_save_interval: 5  # minutes
  notifications:
    enabled: false
    slack_webhook: ""
    email: ""
```

### Environment Variables (.env)

```bash
# API Keys (required)
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
OPENAI_API_KEY=your_openai_key

# HuggingFace (required for HF output)
HUGGINGFACE_TOKEN=your_hf_token

# Optional
SLACK_WEBHOOK_URL=your_slack_webhook
NOTIFICATION_EMAIL=your@email.com
```

---

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/username/data-synthesizer.git
cd data-synthesizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run tests
pytest -m unit -v
```

### GCP Free Tier VM Setup

```bash
# 1. Create VM (e2-micro, 1GB RAM)
gcloud compute instances create synthesizer-vm \
  --machine-type=e2-micro \
  --zone=us-central1-a \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB

# 2. SSH into VM
gcloud compute ssh synthesizer-vm --zone=us-central1-a

# 3. Run setup script
curl -sSL https://raw.githubusercontent.com/username/data-synthesizer/main/scripts/setup_vm.sh | bash

# 4. Configure
cd data-synthesizer
cp .env.example .env
nano .env  # Add your API keys

# 5. Start with screen/tmux (persists after disconnect)
screen -S synth
python scripts/run.py --config config.yaml

# Detach: Ctrl+A, D
# Reattach: screen -r synth
```

---

## Usage

### Command Line

```bash
# Basic run
python scripts/run.py

# With specific config
python scripts/run.py --config configs/contract_generation.yaml

# Dry run (no API calls)
python scripts/run.py --dry-run

# Resume from checkpoint
python scripts/run.py --resume

# Specific synthesis type
python scripts/run.py --type deep_thinking

# Override provider
python scripts/run.py --provider openrouter
```

### Python API

```python
from src.core.config import load_config
from src.synthesizers.qa_synthesis import QASynthesizer
from src.providers.factory import ProviderFactory

# Load configuration
config = load_config("config.yaml")

# Create provider with auto-failover
provider = ProviderFactory.create(config)

# Create synthesizer
synthesizer = QASynthesizer(config, provider)

# Run synthesis
synthesizer.run()
```

### Jupyter Notebook

```python
# In notebooks/qa_synthesis.ipynb

%load_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

from src.core.config import load_config
from src.synthesizers.qa_synthesis import QASynthesizer

config = load_config("../config.yaml")
synthesizer = QASynthesizer(config)

# Show progress
synthesizer.show_progress()

# Continue synthesis
synthesizer.continue_synthesis()
```

---

## Component Specifications

### 1. Provider System

#### Base Provider Interface

```python
class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate content from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass

    @abstractmethod
    def get_usage(self) -> dict:
        """Get token usage statistics."""
        pass
```

#### Provider Factory with Failover

```python
class ProviderFactory:
    @staticmethod
    def create(config: dict) -> BaseProvider:
        """Create provider with automatic failover chain."""
        pass

    def switch_provider(self, reason: str) -> bool:
        """Switch to next provider in failover chain."""
        pass
```

**Tests:**
- `test_provider_creation` - Factory creates correct provider
- `test_provider_failover` - Switches on error
- `test_rate_limiting` - Respects rate limits
- `test_retry_logic` - Retries with backoff

### 2. Progress Tracking

#### Progress Manager

```python
class ProgressManager:
    def __init__(self, config: dict):
        self.progress_file = config['progress_file']
        self.output_handler = config['output_handler']

    def load(self) -> dict:
        """Load progress from persistent storage."""
        pass

    def save(self) -> None:
        """Save progress to persistent storage."""
        pass

    def is_processed(self, item_id: str) -> bool:
        """Check if item already processed."""
        pass

    def mark_processed(self, item_id: str, stats: dict) -> None:
        """Mark item as processed with statistics."""
        pass
```

**Tests:**
- `test_progress_persistence` - Saves and loads correctly
- `test_resume_from_checkpoint` - Resumes after restart
- `test_concurrent_access` - Handles concurrent updates
- `test_statistics_aggregation` - Aggregates stats correctly

### 3. Synthesizers

#### Base Synthesizer

```python
class BaseSynthesizer(ABC):
    def __init__(self, config: dict, provider: BaseProvider):
        self.config = config
        self.provider = provider
        self.progress = ProgressManager(config)
        self.validator = ValidationPipeline(config)
        self.logger = get_logger(__name__)

    @abstractmethod
    def create_prompt(self, item: dict) -> str:
        """Create synthesis prompt for item."""
        pass

    @abstractmethod
    def parse_response(self, response: str) -> list:
        """Parse LLM response into structured data."""
        pass

    def synthesize(self, item: dict) -> list:
        """Main synthesis method with validation."""
        pass

    def run(self) -> None:
        """Run continuous synthesis loop."""
        pass
```

**Tests:**
- `test_prompt_creation` - Creates valid prompts
- `test_response_parsing` - Parses JSON correctly
- `test_validation_pipeline` - Validates output
- `test_error_handling` - Handles API errors gracefully

### 4. Validators

#### Language Validator

```python
class LanguageValidator:
    def __init__(self, target_language: str, min_confidence: float):
        self.target = target_language
        self.threshold = min_confidence

    def validate(self, text: str) -> ValidationResult:
        """Validate text language."""
        pass
```

#### Quality Scorer

```python
class QualityScorer:
    def score(self, original: dict, generated: dict) -> QualityScore:
        """Score generated content quality."""
        pass
```

**Tests:**
- `test_language_detection` - Detects languages correctly
- `test_quality_scoring` - Scores content accurately
- `test_edge_cases` - Handles empty/short text

### 5. Output Handlers

#### HuggingFace Handler

```python
class HuggingFaceOutput:
    def __init__(self, config: dict):
        self.repo_id = config['repository']
        self.token = config['token']

    def save_chunk(self, data: list, chunk_id: str) -> None:
        """Upload chunk to HuggingFace."""
        pass

    def save_progress(self, progress: dict) -> None:
        """Save progress file to repo."""
        pass
```

**Tests:**
- `test_chunk_upload` - Uploads correctly
- `test_progress_sync` - Syncs progress
- `test_retry_on_network_error` - Retries uploads

---

## Roadmap

### Phase 1: Core Infrastructure (MVP)

| Task | Priority | Status |
|------|----------|--------|
| Project structure setup | High | [ ] |
| Configuration system | High | [ ] |
| Logging infrastructure | High | [ ] |
| Base provider interface | High | [ ] |
| Gemini provider | High | [ ] |
| Progress manager | High | [ ] |
| Base synthesizer | High | [ ] |
| QA synthesizer | High | [ ] |
| HuggingFace output | High | [ ] |
| Basic CLI | High | [ ] |
| Unit tests for core | High | [ ] |

### Phase 2: Production Ready

| Task | Priority | Status |
|------|----------|--------|
| OpenRouter provider | Medium | [ ] |
| Provider failover logic | Medium | [ ] |
| Deep thinking synthesizer | Medium | [ ] |
| Corpus generator | Medium | [ ] |
| Language validator | Medium | [ ] |
| Quality scorer | Medium | [ ] |
| Graceful shutdown | Medium | [ ] |
| Health check endpoint | Medium | [ ] |
| Integration tests | Medium | [ ] |
| Jupyter notebooks | Medium | [ ] |

### Phase 3: Enhanced Features

| Task | Priority | Status |
|------|----------|--------|
| OpenAI provider | Low | [ ] |
| Cost tracking | Low | [ ] |
| Notifications | Low | [ ] |
| Web dashboard | Low | [ ] |
| Custom prompt templates | Low | [ ] |
| Multi-instance coordination | Low | [ ] |
| Performance benchmarks | Low | [ ] |

---

## Testing Strategy

### Test Categories

```python
# Unit tests - fast, no external dependencies
@pytest.mark.unit
def test_config_loading():
    pass

# Integration tests - requires API keys
@pytest.mark.integration
def test_gemini_generation():
    pass

# Slow tests - long running
@pytest.mark.slow
def test_full_synthesis_cycle():
    pass
```

### Running Tests

```bash
# Unit tests only (CI-safe)
pytest -m unit -v

# Integration tests (requires API keys)
pytest -m integration -v

# All tests
pytest -v

# With coverage
pytest --cov=src --cov-report=html

# Specific module
python -m src.providers.gemini
```

### Test Coverage Requirements

- **Core modules**: 90%+ coverage
- **Providers**: 80%+ coverage
- **Synthesizers**: 80%+ coverage
- **Validators**: 90%+ coverage

---

## Deployment

### GCP Free Tier Specifications

- **VM**: e2-micro (0.25 vCPU, 1GB RAM)
- **Disk**: 30GB standard
- **Network**: 1GB egress/month free
- **Always Free**: 1 e2-micro instance

### Resource Usage

| Component | Memory | CPU | Notes |
|-----------|--------|-----|-------|
| Python runtime | ~100MB | Minimal | Base overhead |
| Provider client | ~50MB | Minimal | API calls only |
| Logging | ~10MB | Minimal | File-based |
| Progress tracking | ~20MB | Minimal | JSON files |
| **Total** | **~200MB** | **<10%** | Well under 1GB limit |

### Running as Background Service

#### Using Screen

```bash
# Start
screen -S synth
python scripts/run.py
# Detach: Ctrl+A, D

# Reattach
screen -r synth

# List sessions
screen -ls
```

#### Using Systemd

```ini
# /etc/systemd/system/synthesizer.service
[Unit]
Description=Data Synthesizer Service
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/home/your_user/data-synthesizer
ExecStart=/home/your_user/data-synthesizer/venv/bin/python scripts/run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable synthesizer
sudo systemctl start synthesizer

# Check status
sudo systemctl status synthesizer

# View logs
journalctl -u synthesizer -f
```

### Monitoring

```bash
# Check if running
curl http://localhost:8080/health

# View logs
tail -f logs/synthesizer.log

# Check progress
python -c "from src.core.progress import ProgressManager; pm = ProgressManager(); pm.show_progress()"
```

---

## Error Handling

### Provider Errors

| Error Type | Action | Retry |
|------------|--------|-------|
| Rate limit (429) | Switch provider or wait | Yes, with backoff |
| Safety filter | Log and skip item | No |
| Network timeout | Retry with backoff | Yes, 5 attempts |
| Invalid response | Log and retry | Yes, 3 attempts |
| Auth error | Log and stop | No |

### Graceful Shutdown

```python
import signal

def handle_shutdown(signum, frame):
    logger.info("Shutdown signal received")
    progress_manager.save()
    output_handler.flush()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)
```

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Follow code style guidelines
4. Write tests for new features
5. Ensure all tests pass: `pytest -m unit -v`
6. Submit pull request

### Code Style

- Use Black for formatting
- Use type hints
- Add docstrings for public APIs
- Follow existing patterns

---

## License

MIT License - see LICENSE file

---

## Acknowledgments

- Google Gemini API
- OpenRouter
- HuggingFace Hub
- LangID for language detection
