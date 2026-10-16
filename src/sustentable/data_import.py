"""
Not implemented:
- function to read data specific to each type of file.
- function to check if a file is corrupted.
- function to check for a single table in sqlite.

Needs refactor:
- function to check if a file is empty, can be made abstract and be defined for each type of file.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class FileCheck(ABC):
    # Extensions accepted by each concrete check, lowercase and with the leading dot.
    extensions: tuple[str, ...] = ()

    def __init__(self, file_path: Path) -> None:
        # The constructor should receive a Path from `Reader` class constructor
        self.file_path = file_path

    def exists(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File '{self.file_path}' does not exist.")

    def supported_extension(self) -> None:
        ext = self.file_path.suffix
        if ext.lower() not in self.extensions:
            raise ValueError(f"Extension '{ext}' is not supported. Supported extensions: {list(self.extensions)}.")

    def not_empty(self) -> None:
        if self.file_path.stat().st_size == 0 or not self._not_empty():
            raise ValueError(f"File '{self.file_path}' contains no data.")

    @abstractmethod
    def _not_empty(self) -> bool:
        """Checks for each type of file if it has data in it. Called to raise a ValueError."""
        ...

    @abstractmethod
    def not_corrupted(self) -> None:
        # A more specific check can be implemented here depending on the file type.
        # For example, a CSV could be test-read with pandas, a JSON with json.loads, etc.
        pass

    @abstractmethod
    def single_table(self) -> None:
        # A check for SQLite databases can be implemented here.
        # For example, connect to the database and count the number of tables.
        # The implementation of this method for csv and xlsx will remain empty.
        pass


class CSVFileCheck(FileCheck):
    extensions = (".csv",)

    def _not_empty(self) -> bool:
        raise NotImplementedError("Checking for content is not implemented for CSV files.")

    def not_corrupted(self) -> None:
        raise NotImplementedError("Corruption check is not implemented for CSV files.")

    def single_table(self) -> None:
        # Not applicable for CSV files.
        pass


class ExcelFileCheck(FileCheck):
    extensions = (".xlsx", ".xls")

    def _not_empty(self) -> bool:
        raise NotImplementedError("Checking for content is not implemented for Excel files.")

    def not_corrupted(self) -> None:
        raise NotImplementedError("Corruption check is not implemented for Excel files.")

    def single_table(self) -> None:
        # Not applicable for Excel files.
        pass


class SQLiteFileCheck(FileCheck):
    extensions = (".db", ".sqlite", ".sqlite3")

    def _not_empty(self) -> bool:
        raise NotImplementedError("Checking for content is not implemented for SQLite files.")

    def not_corrupted(self) -> None:
        raise NotImplementedError("Corruption check is not implemented for SQLite files.")

    def single_table(self) -> None:
        raise NotImplementedError("Single table check is not implemented for SQLite files.")


class Reader(ABC):
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = file_path

    def read(self) -> None:
        """Validate the file and read its contents."""
        self.validate()
        return self._read()

    @abstractmethod
    def _read(self) -> None:
        """Read the file contents. Called only after ``validate()`` has passed."""
        ...

    @abstractmethod
    def _get_file_check(self) -> FileCheck:
        pass

    def validate(self) -> None:
        """Run every check on the current ``file_path``, raising on the first one that fails."""
        file_check = self.file_check
        file_check.exists()
        file_check.supported_extension()
        file_check.not_empty()
        file_check.not_corrupted()
        file_check.single_table()

    @property
    def file_check(self) -> FileCheck:
        """Checks for the current ``file_path``, rebuilt on every access so they never point to a stale path."""
        return self._get_file_check()

    @property
    def file_path(self) -> Path:
        """Path of the file to check, always stored as a ``Path``."""
        return Path(self._file_path)

    # We should change the value instead of creating a new instance.
    @file_path.setter
    def file_path(self, value: str | Path) -> None:
        if not isinstance(value, (str, Path)):
            raise TypeError("The file path must be a str or a Path.")
        self._file_path = Path(value)


class CSVReader(Reader):
    """Reader for CSV files."""

    def _get_file_check(self) -> FileCheck:
        return CSVFileCheck(self.file_path)

    def _read(self) -> None:
        raise NotImplementedError("Reading is not implemented for CSV files.")


class ExcelReader(Reader):
    """Reader for Excel files (``.xlsx`` and ``.xls``)."""

    def _get_file_check(self) -> FileCheck:
        return ExcelFileCheck(self.file_path)

    def _read(self) -> None:
        raise NotImplementedError("Reading is not implemented for Excel files.")


class SQLiteReader(Reader):
    """Reader for SQLite database files."""

    def _get_file_check(self) -> FileCheck:
        return SQLiteFileCheck(self.file_path)

    def _read(self) -> None:
        raise NotImplementedError("Reading is not implemented for SQLite files.")


_READERS: tuple[tuple[tuple[str, ...], type[Reader]], ...] = (
    (CSVFileCheck.extensions, CSVReader),
    (ExcelFileCheck.extensions, ExcelReader),
    (SQLiteFileCheck.extensions, SQLiteReader),
)


def get_reader(file_path: str | Path) -> Reader:
    """Return the ``Reader`` that matches the extension of ``file_path``."""
    ext = Path(file_path).suffix.lower()
    for extensions, reader_class in _READERS:
        if ext in extensions:
            return reader_class(file_path)
    raise ValueError(f"Extension '{ext}' is not supported.")
