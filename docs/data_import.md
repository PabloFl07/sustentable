# Data import module

`sustentable.dataset.readers` is the module in charge of loading the data that the regression models are built from. The user points the application to a file, and the module decides how to read it, checks that it is usable and returns its contents. Three formats are supported for now: CSV (`.csv`), Excel (`.xlsx` and `.xls`) and SQLite databases (`.db`, `.sqlite` and `.sqlite3`). Extensions are compared without regard to case, so a file called `DATA.CSV` is treated exactly like `data.csv`.

> The module is still a skeleton and cannot read any file yet. See [Current status](#current-status) before relying on it.

## Quick start

Most callers only need two lines. `get_reader()` picks the right reader from the file extension, and `read()` validates the file and loads it:

```python
from sustentable.dataset.readers import get_reader

reader = get_reader("data/emissions.csv")
data = reader.read()
```

If anything is wrong with the file, one of these two calls raises an exception with a message that can be shown to the user as it is. This is covered in more detail in [Reading a file from the interface](#reading-a-file-from-the-interface).

## Design

The module is built around two small class hierarchies. The `FileCheck` classes know how to tell whether a file is valid, and the `Reader` classes know how to load it. Every format has one class of each kind, and each reader creates the check for its own format:

```
FileCheck (ABC)                Reader (ABC)
├── CSVFileCheck     <──────── ├── CSVReader
├── ExcelFileCheck   <──────── ├── ExcelReader
└── SQLiteFileCheck  <──────── └── SQLiteReader
```

Keeping validation and reading apart means that the checks can be tested on their own, and that a reader can assume it is working with a valid file by the time it starts parsing.

### File checks

`FileCheck` receives a `Path` and exposes one method per check. Each method returns nothing when the check passes and raises an exception when it fails, so they can be called one after another without inspecting any return value.

Three of the checks are shared by every format and live in the base class. `exists()` raises `FileNotFoundError` if there is nothing at the given path, `supported_extension()` raises `ValueError` if the extension is not one of the class's `extensions`, and `not_empty()` raises `ValueError` if the file is zero bytes long. Because the extension check uses the extensions of the class it belongs to, a CSV check given an `.xlsx` file fails, even though `.xlsx` is supported by the module as a whole.

The other two checks depend on the format, so they are abstract. `not_corrupted()` should make sure that the file can actually be parsed, for example by trying to open it with the library that will later read it. `single_table()` only matters for SQLite, where the database must contain exactly one table so the module knows which one to load; the CSV and Excel checks implement it as a no-op.

The checks can be used directly, which is handy in tests:

```python
from pathlib import Path

from sustentable.dataset.readers import CSVFileCheck

check = CSVFileCheck(Path("data/emissions.xlsx"))
check.exists()  # passes if the file is there
check.supported_extension()  # ValueError: Extension '.xlsx' is not supported. Supported extensions: ['.csv'].
```

### Readers

`Reader` is the base class for every reader. It is created with the path of the file, which can be passed as a `str` or a `Path` and is always stored as a `Path`. Any other type raises `TypeError`. The path can be changed later on the same instance, and the `file_check` property builds a new check on every access, so the reader never validates a path it no longer points to:

```python
from sustentable.dataset.readers import CSVReader

reader = CSVReader("data/2024.csv")
reader.file_path = "data/2025.csv"
reader.file_check.file_path  # PosixPath('data/2025.csv')
```

`validate()` runs the checks in a fixed order (existence, extension, emptiness, corruption and, finally, the single table check) and stops at the first one that fails. The order matters: there is no point in checking the size of a file that does not exist, or in trying to parse a file with the wrong extension.

`read()` is the public way to load data. It calls `validate()` and, only if every check passes, calls `_read()` and returns its result. Subclasses must not override `read()`, because doing so would let them skip validation.

A concrete reader therefore only has to implement two methods. `_get_file_check()` returns the `FileCheck` for its format, and `_read()` does the actual reading. Since `_read()` is only ever called after a successful validation, it does not need to repeat any of those checks.

### Choosing a reader

`get_reader(file_path)` looks at the extension of the path and returns an instance of the matching reader. If no reader handles that extension it raises `ValueError`. The function does not keep its own list of extensions; it takes them from the `FileCheck` classes through the `_READERS` table, so each extension is declared in exactly one place.

```python
>>> get_reader("data/emissions.sqlite")
<sustentable.dataset.readers.SQLiteReader object at 0x...>
>>> get_reader("data/emissions.json")
ValueError: Extension '.json' is not supported.
```

## Reading a file from the interface

When the user types a path in the interface, it arrives as a plain `str`. That string goes straight into `get_reader()`, which either returns a reader or rejects the extension. The interface then calls `read()` on the reader, which validates the file and loads it.

Every failure along the way is reported through an exception whose message is written for the user, so the interface does not need to know which check failed. It only has to catch the exceptions, show the message and let the user fix the path or the file and try again:

```python
from sustentable.dataset.readers import get_reader


def load_from_user_input(path: str):
    try:
        reader = get_reader(path)
        return reader.read()
    except (FileNotFoundError, ValueError) as error:
        show_error(str(error))  # whatever the interface uses to display messages
        return None
```

Once `not_corrupted()` and `single_table()` are implemented they may raise other exception types depending on the library used to open each format. When that happens, either those checks should translate their errors into `ValueError`, or the `except` clause above has to be extended.

## Adding a new format

Supporting a new format takes three steps. First, create a `FileCheck` subclass that declares its `extensions` and implements `not_corrupted()` and `single_table()`. Then create a `Reader` subclass that returns that check from `_get_file_check()` and implements `_read()`. Finally, register the pair in `_READERS` so that `get_reader()` can find it.

## Current status

Up to 2026-10-16, the structure of the module is in place, but the parts that touch the content of the files are not. `not_corrupted()` raises `NotImplementedError` in all three checks, as does `single_table()` in `SQLiteFileCheck`, and `_read()` is not implemented in any of the readers. Because `validate()` runs `not_corrupted()`, both `validate()` and `read()` currently fail even for perfectly valid files. The return type of `read()` is also still `None`, and will have to change once the readers return real data.
