# Development Workflow

This document describes the AI-assisted development workflow used to build and maintain this project. **Use this as a prompt template for future projects.**

## Overview

This project uses Claude Code as an AI development assistant to:
- Implement features and refactor code
- Write and fix tests
- Manage Git operations
- Ensure CI/CD compliance
- Maintain code quality and alignment

---

## Project Definition Methodology

When starting a new project, follow this structured approach:

### 1. Define Architecture First

Before writing code, establish the system architecture:

```markdown
## System Architecture Template

### Core Components
1. **Data Layer** - How data flows in/out
2. **Processing Layer** - Business logic
3. **Interface Layer** - User interaction points
4. **Infrastructure** - Models, configs, utilities

### Data Flow
Query → Detection → Processing → Generation → Response

### Component Relationships
- Define which components depend on others
- Identify shared resources (models, configs)
- Plan for lazy loading to avoid import issues
```

### 2. Define Feature Roadmap

Organize features by deployment priority:

```markdown
## Feature Prioritization

### Minimum Viable Product (MVP)
Required for first deployment:
- [ ] Core functionality (search, retrieval)
- [ ] Basic API endpoint
- [ ] Configuration management
- [ ] Error handling and logging

### Phase 1: Production Ready
- [ ] Complete test coverage
- [ ] CI/CD pipeline
- [ ] Docker deployment
- [ ] Documentation

### Phase 2: Enhanced Features
- [ ] User interface (Gradio/Streamlit)
- [ ] Multiple providers
- [ ] Caching and optimization
- [ ] Analytics

### Phase 3: Advanced
- [ ] Multi-database support
- [ ] Advanced analytics
- [ ] Collaborative features
```

### 3. Define Directory Structure

Plan the codebase organization:

```markdown
## Directory Structure Guidelines

### Root Level
- config.py          # All settings in one place
- main.py            # Single entry point
- requirements.txt   # Dependencies
- Dockerfile         # Deployment

### Module Organization
project/
├── core/            # Business logic
│   ├── search/      # Retrieval components
│   └── generation/  # Output components
├── providers/       # External integrations
├── pipeline/        # High-level orchestration
├── api/             # REST endpoints
├── ui/              # User interfaces
└── tests/           # Test infrastructure

### Naming Conventions
- Modules: lowercase_with_underscores
- Classes: PascalCase
- Functions: lowercase_with_underscores
- Constants: UPPERCASE_WITH_UNDERSCORES
```

### 4. Define Minimum Deployment Requirements

```markdown
## Deployment Checklist

### Must Have
- [ ] All unit tests passing
- [ ] No import errors in CI
- [ ] Environment variable documentation
- [ ] Health check endpoint
- [ ] Graceful error handling

### Should Have
- [ ] Docker support
- [ ] CI/CD pipeline
- [ ] API documentation
- [ ] Performance benchmarks

### Nice to Have
- [ ] Kubernetes configs
- [ ] Monitoring/alerting
- [ ] Auto-scaling
```

---

## Code Organization Patterns

### Lazy Import Pattern

Prevent import errors during CI testing:

```python
# In __init__.py
def __getattr__(name):
    if name == 'HeavyClass':
        from .heavy_module import HeavyClass
        return HeavyClass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['HeavyClass']
```

### Singleton Pattern

For shared resources:

```python
_instance = None

def get_instance():
    global _instance
    if _instance is None:
        _instance = MyClass()
    return _instance
```

### Factory Pattern

For pluggable components:

```python
PROVIDERS = {
    'local': LocalProvider,
    'api': APIProvider,
}

def create_provider(name, config=None):
    return PROVIDERS[name](config)
```

### Runnable Test Blocks

Add to every module for direct testing:

```python
if __name__ == "__main__":
    print("=" * 60)
    print("MODULE TEST")
    print("=" * 60)

    # Test code here
    instance = MyClass()
    result = instance.test_method()
    print(f"  ✓ Test passed: {result}")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
```

---

## Hardware and Resource Management

### Multi-GPU Detection

```python
@dataclass
class HardwareConfig:
    embedding_device: str      # 'cpu' or 'cuda:N'
    llm_device: str
    quantization: str          # 'none', '4bit', '8bit'
    device_map: Dict[str, int] # component -> gpu_index

def detect_hardware() -> HardwareConfig:
    # Auto-detect and distribute workloads
    pass
```

### Provider Abstraction

```python
class BaseProvider(ABC):
    @abstractmethod
    def initialize(self) -> bool: pass

    @abstractmethod
    def generate(self, prompt: str) -> str: pass

    @abstractmethod
    def shutdown(self) -> None: pass
```

---

## Testing Strategy

### Test Categories

```python
# Unit tests - no external dependencies
@pytest.mark.unit
def test_query_detection():
    pass

# Integration tests - requires models
@pytest.mark.integration
def test_full_pipeline():
    pass

# Skip if dependency missing
numpy = pytest.importorskip("numpy")
```

### Test File Structure

```
tests/
├── unit/           # Fast, no GPU
├── integration/    # Requires models
└── conftest.py     # Shared fixtures
```

### Running Tests

```bash
# Unit only (CI)
pytest -m unit -v

# Integration (local with GPU)
pytest -m integration -v

# Specific module
python -m module_name  # Uses __main__ block
```

---

## Branch Strategy

### Feature Branches
- All development happens on feature branches: `claude/<feature-name>-<session-id>`
- Never push directly to `main` or `master`
- Each session gets a unique branch for tracking

### Git Operations
```bash
# Always use -u flag when pushing
git push -u origin claude/<branch-name>

# Commit with descriptive messages using HEREDOC
git commit -m "$(cat <<'EOF'
Short summary line

- Detailed change 1
- Detailed change 2
EOF
)"
```

---

## Development Cycle

### 1. Task Planning
- Break complex tasks into smaller, trackable items
- Use todo lists to track progress
- Prioritize based on dependencies

### 2. Implementation
- Follow existing code patterns
- Use lazy imports to avoid heavy dependencies during testing
- Prefer editing existing files over creating new ones

### 3. Testing
- Run tests frequently: `pytest -m unit -v`
- Fix failing tests before committing
- Use `pytest.importorskip()` for optional dependencies
- Mark tests appropriately:
  - `@pytest.mark.unit` - No external dependencies
  - `@pytest.mark.integration` - Requires models/GPU

### 4. CI Compliance
- Ensure all unit tests pass locally before pushing
- Use lazy imports in `__init__.py` files to prevent import errors
- Check that requirements.txt has all dependencies uncommented

### 5. Code Review
- Verify alignment with original code (e.g., Kaggle_Demo.ipynb)
- Check that new features integrate properly
- Ensure backward compatibility

---

## Common Commands

### Testing
```bash
# Run all unit tests
pytest -m unit -v

# Run specific test file
pytest tests/unit/test_providers.py -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run module directly
python -m core.analytics
python -m providers.factory
python -m conversation.context_cache
python -m hardware_detection
```

### Git
```bash
# Check status
git status

# View recent commits
git log --oneline -5

# Stage and commit
git add -A && git commit -m "message"

# Push to remote
git push -u origin <branch-name>
```

### Development
```bash
# Run Gradio UI
python ui/gradio_app.py

# Run FastAPI server
python api/server.py

# Run CLI
python main.py --query "your question"

# Hardware detection
python hardware_detection.py
```

---

## CI/CD Pipeline

### GitHub Actions Workflow
1. **Test Job** - Runs unit tests on Python 3.9, 3.10, 3.11
2. **Lint Job** - Checks formatting with Black, isort, flake8
3. **Build Job** - Creates distribution package
4. **Docker Job** - Builds and tests Docker image (on main only)

### CI Requirements
- All unit tests must pass
- No critical linting errors (E9, F63, F7, F82)
- Package must build successfully

---

## Troubleshooting

### Import Errors in CI
**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Use lazy imports in `__init__.py`:
```python
def __getattr__(name):
    if name == 'HeavyClass':
        from .heavy_module import HeavyClass
        return HeavyClass
    raise AttributeError(...)
```

### Test Failures
**Problem**: Tests fail due to missing dependencies

**Solution**: Add skip markers:
```python
import pytest
numpy = pytest.importorskip("numpy")
```

### API Mismatches
**Problem**: Tests use wrong method names

**Solution**: Check actual implementation and update tests:
```python
# Wrong: detector.detect(query)
# Right: detector.analyze_query(query)
```

---

## Best Practices

### Code Quality
- Follow existing patterns in the codebase
- Add type hints for function signatures
- Include docstrings for public APIs
- Log important operations with `logger_utils`

### Testing
- Test one thing per test function
- Use descriptive test names
- Include both positive and edge cases
- Mock external dependencies

### Documentation
- Update README when adding features
- Document configuration options in .env.example
- Keep directory structure map current

### Git Commits
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Include details in commit body
- Reference issues when applicable

---

## Session Workflow Example

```
1. User: "Add feature X"

2. Claude: Creates todo list
   - [ ] Research existing code
   - [ ] Implement feature
   - [ ] Write tests
   - [ ] Update documentation

3. Claude: Implements feature, marking todos as complete

4. Claude: Runs tests
   $ pytest -m unit -v

5. Claude: Fixes any failures

6. Claude: Commits and pushes
   $ git add -A
   $ git commit -m "Add feature X"
   $ git push -u origin claude/branch-name

7. User: Reviews and provides feedback

8. Repeat until complete
```

---

## Prompt Template for New Projects

Use this template when starting a new AI-assisted project:

```markdown
# Project: [Name]

## Architecture
[Define high-level architecture with ASCII diagrams]

## Features by Priority

### MVP (Required for deployment)
- [ ] Feature 1
- [ ] Feature 2

### Phase 1 (Production ready)
- [ ] Feature 3
- [ ] Feature 4

### Phase 2 (Enhanced)
- [ ] Feature 5

## Directory Structure
```
project/
├── core/           # Business logic
├── api/            # Endpoints
├── ui/             # Interface
└── tests/          # Testing
```

## Deployment Requirements
- [ ] Unit tests passing
- [ ] Docker support
- [ ] Environment docs

## Conventions
- Use lazy imports for heavy dependencies
- Add __main__ blocks for direct testing
- Follow existing patterns
```

---

## Alignment Verification

When refactoring from monolithic code (like Jupyter notebooks):

1. **List original components**
   ```bash
   grep "^class \|^def " original.py
   ```

2. **Map to modular structure**
   - Original `ClassName` → `module/class_name.py`

3. **Verify functionality**
   - Check method signatures match
   - Ensure return types are compatible
   - Test with same inputs

4. **Document differences**
   - New features not in original
   - Improved implementations
   - Breaking changes

---

## Contributing

1. Create feature branch from main
2. Follow development cycle above
3. Ensure CI passes
4. Submit PR with description
5. Address review feedback
6. Merge when approved
