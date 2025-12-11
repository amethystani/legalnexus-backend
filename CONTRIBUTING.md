# Contributing to LegalNexus

Thank you for your interest in contributing to LegalNexus! This document provides guidelines for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Be kind and considerate to other contributors.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/legalnexus-backend.git
   cd legalnexus-backend
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/amethystani/legalnexus-backend.git
   ```

## How to Contribute

### Reporting Bugs

- Use the [Bug Report](https://github.com/amethystani/legalnexus-backend/issues/new?template=bug_report.md) template
- Include reproduction steps
- Provide environment details

### Suggesting Features

- Use the [Feature Request](https://github.com/amethystani/legalnexus-backend/issues/new?template=feature_request.md) template
- Explain the use case
- Describe expected behavior

### Submitting Code

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Test your changes:
   ```bash
   python src/evaluation/real_evaluation.py
   pytest tests/
   ```
4. Commit with a descriptive message:
   ```bash
   git commit -m "Add: brief description of changes"
   ```
5. Push and create a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r config/requirements.txt

# Install development dependencies
pip install pytest flake8

# Pull LFS files
git lfs pull
```

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Type hints are encouraged

### File Organization

| Type | Location |
|------|----------|
| Core algorithms | `src/core/` |
| Evaluation scripts | `src/evaluation/` |
| UI components | `src/ui/` |
| Utilities | `src/utils/` |
| Scripts/tools | `scripts/tools/` |
| Tests | `tests/` |

### Commit Messages

Use the following prefixes:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for updates to existing features
- `Docs:` for documentation changes
- `Refactor:` for code refactoring
- `Test:` for test additions/changes

## Pull Request Process

1. Ensure your code passes all tests
2. Update documentation if needed
3. Fill out the PR template completely
4. Request review from maintainers
5. Address any feedback
6. Once approved, maintainers will merge

## Questions?

Open an issue or contact the maintainers.

Thank you for contributing!
