# Agent instructions

Follow `CONTRIBUTING.md`. These rules win when they conflict with a general coding style.

## Scope

- Change only what the current task requires. Leave unrelated files alone.
- Do not edit `README.md` unless the task explicitly says to.
- Do not add dependencies or edit `pyproject.toml` / `uv.lock` unless the task needs a dependency change. Use `uv add`, not pip.
- Do not create markdown files, docs, or a changelog unless asked.
- Do not un-ignore or commit `data/`, `work/`, or `.historias/`.

## Git

- Do not commit, push, merge, rebase, or amend unless the user explicitly asks.
- Do not commit on `main` or `dev`. Branch from an up-to-date `dev`:
  `feature/<n>-<description>`, `fix/<n>-<description>`, or `docs/<description>`.
- Do not force-push, skip hooks (`--no-verify`), or rewrite history.
- Stage files by name. Do not run `git add .` or `git add -A`.
- Commits use Conventional Commits: `type(scope): imperative summary`
  (imperative, no period, 72 characters max).

## Never stage

`.env`, credentials, `data/`, datasets, `.pkl`, `.joblib`, `.venv/`, caches, and editor folders. The full list is in `CONTRIBUTING.md` and `.gitignore`.

## Code

- Python >= 3.14. Application code goes in `src/sustentable/`. Tests go in `tests/`.
- Ruff is the formatter and linter. Line length is **144** (`pyproject.toml`). Do not reformat to 88.
- `snake_case` for functions, variables, and modules; `PascalCase` for classes; `UPPER_CASE` for constants.
- Type hints on every public function. Docstrings on every public module, class, and function.
- Code, comments, and docstrings in English.
- Keep the entry point `sustentable = "sustentable:main"`.

## Before finishing

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```
