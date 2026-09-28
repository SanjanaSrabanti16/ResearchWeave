# Contributing to ResearchWeave

ResearchWeave is an active research prototype. Small, focused changes with explicit tests are easiest to review.

## Development setup

Use Python 3.11 and Node.js 20.19+ (or 22.13+). From a fresh clone:

```bash
cd backend
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -e ".[dev]"

cd ../frontend
npm ci
```

Copy `.env.example` to `.env` only if you need local overrides. Basic tests do not need scholarly-provider or LLM credentials.

## Quality checks

Run before opening a pull request:

```bash
cd backend
pytest
ruff check .
ruff format --check .

cd ../frontend
npm test
npm run lint
npm run typecheck
npm run build
```

When container files change, also run `docker compose config --quiet` and `docker compose build backend frontend` from the repository root.

## Pull requests

- Explain the user-visible or architectural effect and keep the diff limited to that purpose.
- Add focused regression tests for behavior changes and update relevant documentation.
- Preserve the frozen retrieval, canonical identity, ranking, evidence-validation, and graph semantics unless a proposal explicitly changes them with scientific justification, tests, and documentation.
- Do not commit `.env` files, API keys, tokens, PDFs, parsed-paper output, model files, local databases, caches, logs, or provider response dumps.
- Do not make tests depend on live scholarly APIs, hosted LLMs, local Ollama, or downloaded models when a deterministic fake can establish the behavior.

## Issues

Bug reports should include reproduction steps, expected and observed behavior, platform/runtime versions, and sanitized logs. Never include credentials or private paper contents. Use the private process in [SECURITY.md](SECURITY.md) for vulnerabilities.
