# Contribution Guide

This guide explains how we work as a team. Please read it before your first contribution; following it keeps the history clean, avoids conflicts and makes reviews faster for everyone.

## Index

- [Setup](#setup)
- [Workflow](#workflow)
- [Commits](#commits)
- [Branch model](#branch-model)
- [Style guide](#style-guide)
- [What not to commit](#what-not-to-commit)
  - [Handling secrets](#handling-secrets)
- [Code testing](#code-testing)
- [Documentation](#documentation)
- [Use of AI](#use-of-ai)


## Setup

The only tools you need are [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/).
uv installs the right Python version and all project dependencies for you, so there is no need to install Python manually or use pip.
All dependencies are listed in `pyproject.toml`.

> [!NOTE]
> If you cannot use uv, a `requirements.txt` file is also provided as an alternative.

```bash
git clone https://github.com/PabloFl07/sustentable.git && cd sustentable
uv sync
uv run pre-commit install
uv run sustentable
uv run pytest
```

## Workflow

#### Notes
- Correct [commit messages](#commits) and [branch names](#branch-model)
- What is allowed to commit: [What not to commit](#what-not-to-commit)

**1. Create a new branch based on `dev`**

> [!IMPORTANT]
> Always pull the latest `dev` before creating a new branch or merging one to avoid conflicts.


```bash
git switch dev
git pull
git switch -c correctly-named-branch
```

**2. Work on it and push it**

```bash
git add ...
git commit -m "Correctly redacted commit message"
git push -u correctly-named-branch
```

**3. Switch to dev and merge your branch**

```bash
git switch dev && git pull
git merge correctly-named-branch
git push
```

#### Tips

**1. Reuse a previous branch**

```bash
git switch previous-branch
git rebase dev        # Pulls the actual dev status to the branch | Works as if you created it from dev.
```
Now yo can get more work done on that branch without causing any conflicts.


## Commits

We follow [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/):

```
<type>(<optional scope>): <description>

[optional body]

[optional footer]
```

- The description must be written in the imperative, without a period, and no more than 72 characters long.
- The `scope` is optional. If you include it, use the name of the affected module (`import`, `gui`, `model`, `storage`, etc.).


|Type|When|
|---|---|
|`feat`|	Adds a new feature or capability for the user, such as a new screen, option, command or supported input.
|`fix`|	Fixes a bug, meaning behavior that didn’t match what was expected. Includes crashes, wrong results and unhandled edge cases.
|`docs`|	Changes only documentation: README, docstrings, code comments, guides or usage examples. No code behavior changes.
|`refactor`|	Restructures or cleans up code without changing what it does: renaming, splitting functions, moving files or removing duplication.
|`test`|	Adds new tests, or fixes or updates existing ones, without touching production code.
|`build`|  Changes the build system or dependencies: adding, removing or upgrading packages, editing pyproject.toml, the uv lockfile or packaging settings.
|`ci`|	Changes continuous integration configuration, such as GitHub Actions workflows that run tests, linters or deployments automatically.
|`chore`|	Routine maintenance that doesn’t affect the code’s behavior and doesn’t fit any other type, such as updating .gitignore, editor config or repository housekeeping.


> [!Tip]
> Append ! after the type/scope when the commit introduces a `BREAKING CHANGE`

## Branch Model

No developer commits directly to `main` or `dev`; all work is done on `feature`, `fix` or `docs` branches.
The number corresponds to the task created in Taiga.

| Branch | Usage |
|---|---|
| `main` | Released versions only.  |
| `dev` | Integration of finished work **via merge exclusively**.  |
| `feature/<nº>-<description>` | One branch per user story or task.  |
| `fix/<nº>-<description>` | Any fix applied in the proyect.  |
| `docs/<description>` | Documentation changes or additions.|

`main` contains only stable versions and only receives changes from `dev` via merge

`dev` branches off `main` and is merged back into `main`

`type branches` branch off `dev` and are merged back into `dev` upon completion of a history or task.

> [!IMPORTANT]
> **Branch deletion criteria**
>
> You should push the branches you're working on to the remote repository. If you need to make a small change, use an existing branch that best fits the change.

Check the [tips](#tips)

## Style guide

All code must follow Python's [PEP 8](https://peps.python.org/pep-0008/) style guide. We enforce it with [Ruff](https://docs.astral.sh/ruff/), which acts as both linter and formatter. Its configuration lives in `pyproject.toml`.

You can check and format your coude manually:


```bash
uv run ruff format .        # format the code
uv run ruff check . --fix   # lint and fix what can be fixed automatically
```

However, there are pre-commit hooks configured that run these checks automatically.

#### Additional conventions

- **Naming:** `snake_case` for variables, functions and modules; `PascalCase` for classes; `UPPER_CASE` for constants.
- Add **Type hints:**  to every public function's parameters and return value.
- **Docstrings:** every public module, class and function must have one.
- **Imports:** grouped as standard library, third-party and local, in that order. Ruff sorts them automatically.
- **Language:** code, comments and docstrings are written in English.

## What not to commit

Only commit what is needed to understand, build and run the project. Anything that can be generated, downloaded or is private stays out of the repository.


| Examples | Why |
|---|---|
| `.env`, API keys, tokens, passwords, credentials | Anyone with access to the repository, or its history, could use them. |
| `data/`, large `.csv`, `.parquet`, images | They bloat the repository and are often licensed or private. |
| `.pkl`, `.joblib`, `.pt`, `.h5` | Large binary files that can be regenerated by training. |
| `.venv/` | `uv sync` recreates it from `uv.lock`. |
| `__pycache__/`, `.ruff_cache/`, `.pytest_cache/`, logs, outputs | Created automatically when running the code. |
| `.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db` | Specific to each developer's editor or OS. |

These are already covered by `.gitignore`. Check `git status` before every commit to make sure nothing unexpected is staged, and avoid `git add .` unless you have reviewed the changes.

> [¡TIP]
>
> Pre-commit hooks also check for any of these files

### Handling secrets

Store secrets in a local `.env` file, which is ignored by Git, and keep an up-to-date `.env.example` with the variable names and placeholder values:


> [!CAUTION]
> **If a secret is committed by mistake**
>
> Deleting it in a new commit is not enough, because it remains in the history. Revoke or regenerate the key immediately and notify the rest of the team.


## Code testing


This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for dependency management and `pytest` for testing. Make sure you have `uv` installed.

### Install dependencies

Install the project dependencies, including the development dependencies:

```bash
uv sync --all-groups
```

### Run the tests

Run the test suite using `pytest` through `uv`:


```bash
uv run pytest
```

### Run specific tests

To run tests from a specific file:


```bash
uv run pytest tests/test_example.py
```

To run a specific test function:


```bash
uv run pytest tests/test_example.py::test_example
```

To display detailed output for each test:

```bash
uv run pytest -v
```

### Test coverage

Please ensure that all tests pass before submitting a pull request. If you introduce new functionality or fix a bug, add or update tests accordingly.

## Documentation

All forms of documentations must be written in Markdown format.

- `README.md`: basic information about the project. You may not modify it
- `AGENTS.md`: rules for coding agents
- `docs/`: detailed documentation (architecture, design decisions, user manual)
- `CHANGELOG.md`: brief description of the changes in each version

We place great value on code comments. Poorly commented code may be rejected.

## Use of AI

We support using AI (LLMs and coding agents) as tools. You remain responsible for every change you publish.

If you use an agent, it must follow [AGENTS.md](AGENTS.md). Before you commit, read the full diff, drop anything outside the task, and run Ruff and pytest yourself.
