[![Python 3.14](https://img.shields.io/badge/python-3.14-orange.svg)](https://www.python.org/downloads/)

# SUSTENTABLE

This project is a Python application that allows you to create, visualize, and predict linear regression models (simple and multiple) from data stored in CSV files, Excel spreadsheets, and SQLite databases. The application includes a graphical user interface (GUI) for ease of use, as well as features to save, load, and utilize trained models.

---

### Prerequisites
- Python 3.14 or higher
- uv 0.13 or higher

### Steps to run the application
1. Clone the repository:
   ```bash
   git clone https://github.com/PabloFl07/sustentable.git
   ```
2. Navigate to the project directory:
   ```bash
   cd sustentable
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```

---

### Key Features

1. **Train Model**
   - Select dependent/independent variables.
   - Choose between simple or multiple regression.
   - Visualize results (R² score, coefficients, graphs...).

2. **Save/Load Model**
   - Save models in `pickle` or `joblib` format.
   - Load previously saved models for quick predictions.

3. **Make Predictions**
   - Input new values and get predictions from the model.

---

## Code Style Guidelines
- We follow **PEP8** style guidelines for Python.
- Variable and function names in `snake_case`.
- Comments and documentation using docstrings.

---

## Repository Structure
```
regression-app/
│
├── src/
│   └── sustentable/
│       ├── __init__.py               # Entry point: main()
│       ├── dataset/                  # Data loading, no GUI dependencies
│       │   ├── readers.py            # File checks and readers (CSV, Excel, SQLite)
│       │   └── exceptions.py         # Data import errors
│       └── gui/                      # Pysual windows and dialogs
│           ├── dataset_window.py     # Window that loads and shows a dataset
│           ├── error_dialog.py       # Modal error messages
│           └── controls/             # Reusable Pysual controls
│               ├── panda_grid.py     # DataFrame grid
│               └── loading_spinner.py
├── tests/                # Unit and integration tests, mirroring src/sustentable/
│   ├── conftest.py       # Example data files shared by all tests
│   ├── dataset/
│   └── gui/
├── docs/                 # Module documentation
├── prototipos/           # Proofs of concept
├── AGENTS.md             # Rules for coding agents
├── CONTRIBUTING.md       # Contributing guide
└── README.md             # This file
```

---

## Contribution & Collaboration

To contribute in this project, please read first the [contribution guide](CONTRIBUTING.md). If you use a coding agent, it must follow [AGENTS.md](AGENTS.md).

---

*This project was developed using Scrum with an Agile approach.*
