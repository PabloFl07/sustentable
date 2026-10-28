"""Read a data file (CSV, Excel or SQLite) into a ``DataFrame`` after checking that it can be imported.

Responsibilities are split in two: a ``FileCheck`` judges and a ``Reader`` reads. A ``FileCheck`` never opens the file
with pandas or sqlite3; it only looks at the path, or at data the ``Reader`` hands to it. A ``Reader`` does all the
parsing, so it is the one that turns the errors of its library into ``CorruptedFileError``.

``Reader.validate()`` runs the checks that need no parsing (the path exists, is a readable, non-empty file and has a
supported extension). ``Reader.read()`` runs them, parses the file once and checks that the result has data. Every
error derives from ``DataImportError`` (see ``sustentable.core.exceptions``).
"""

import sqlite3 as sql
from abc import ABC, abstractmethod
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from openpyxl.utils.exceptions import InvalidFileException
from pandas import DataFrame, errors, read_csv, read_excel, read_sql_query
from xlrd.biffh import XLRDError

from sustentable.core.exceptions import (
    CorruptedFileError,
    EmptyFileError,
    MissingFileError,
    MultipleTablesError,
    NotAFileError,
    UnreadableFileError,
    UnsupportedExtensionError,
)

# User tables of a SQLite database, leaving out the internal ones such as sqlite_sequence.
_SQLITE_TABLES = "FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"


class FileCheck:
    """Rules a file must follow. They look at the path or at data already read, never at the file contents directly.

    Not abstract: every rule is shared, and each format only sets its ``extensions`` or adds rules of its own.
    """

    # Extensions accepted by each concrete check, lowercase and with the leading dot.
    extensions: tuple[str, ...] = ()

    def __init__(self, file_path: Path) -> None:
        # The constructor should receive a Path from `Reader` class constructor
        self.file_path = file_path

    def exists(self) -> None:
        """Raise ``MissingFileError`` if the path does not point to anything."""
        if not self.file_path.exists():
            raise MissingFileError(f"File '{self.file_path}' does not exist.")

    def is_file(self) -> None:
        """Raise ``NotAFileError`` if the path is not a regular file, such as a directory with a supported extension."""
        # NotAFileError instead of IsADirectoryError so the interface can catch it like the other user-facing errors.
        if not self.file_path.is_file():
            raise NotAFileError(f"'{self.file_path}' is not a file.")

    def readable(self) -> None:
        """Raise ``UnreadableFileError`` if the user has no permission to read the file."""
        # Opening the file is more reliable than os.access(), which checks the real user instead of the effective one
        # and can be wrong with ACLs or network file systems. It is also what pandas and sqlite3 will do later.
        try:
            with self.file_path.open("rb"):
                pass
        except PermissionError as e:
            raise UnreadableFileError(f"File '{self.file_path}' cannot be read: permission denied.") from e

    def supported_extension(self) -> None:
        """Raise ``UnsupportedExtensionError`` if the extension is not one of ``extensions``."""
        ext = self.file_path.suffix
        if ext.lower() not in self.extensions:
            raise UnsupportedExtensionError(f"Extension '{ext}' is not supported. Supported extensions: {list(self.extensions)}.")

    def not_zero_bytes(self) -> None:
        """Raise ``EmptyFileError`` if the file has zero bytes, which some libraries would report as corrupted."""
        if self.file_path.stat().st_size == 0:
            raise EmptyFileError(f"File '{self.file_path}' contains no data.")

    def not_empty(self, df: DataFrame | None) -> None:
        """Raise ``EmptyFileError`` if ``df``, the data read from the file, has no rows."""
        if df is None or df.empty:
            raise EmptyFileError(f"File '{self.file_path}' contains no data.")


class CSVFileCheck(FileCheck):
    """Rules for CSV files."""

    extensions = (".csv",)


class ExcelFileCheck(FileCheck):
    """Rules for Excel files (``.xlsx`` and ``.xls``)."""

    extensions = (".xlsx", ".xls")


class SQLiteFileCheck(FileCheck):
    """Rules for SQLite database files."""

    extensions = (".db", ".sqlite", ".sqlite3")

    def single_table(self, tables: list[str]) -> None:
        """Raise ``EmptyFileError`` if ``tables``, the user tables of the database, is empty, or ``MultipleTablesError`` if it has several."""
        if not tables:
            raise EmptyFileError(f"File '{self.file_path}' contains no tables.")
        if len(tables) > 1:
            raise MultipleTablesError(f"File '{self.file_path}' contains {len(tables)} tables, but only one can be imported.")


class Reader(ABC):
    """Validate and read one file. Each subclass parses its format and provides the ``FileCheck`` with its rules."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = file_path

    def read(self) -> DataFrame:
        """Validate the file and return its contents, raising a ``DataImportError`` on the first problem found."""
        self.validate()
        df = self._parse()
        self.file_check.not_empty(df)
        return df

    def validate(self) -> None:
        """Run the checks that need no parsing. Corrupted or empty content is only found by ``read()``."""
        file_check = self.file_check
        file_check.exists()
        file_check.is_file()
        file_check.readable()
        file_check.supported_extension()
        file_check.not_zero_bytes()

    @abstractmethod
    def _parse(self) -> DataFrame | None:
        """Read the whole file with the library of its format, raising ``CorruptedFileError`` if it cannot be parsed.

        Return ``None`` if the file holds nothing to parse. Called only after ``validate()`` has passed.
        """

    @abstractmethod
    def _get_file_check(self) -> FileCheck:
        """Return the ``FileCheck`` of this format for the current ``file_path``."""

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
    """Reader for CSV files encoded in UTF-8."""

    def _get_file_check(self) -> CSVFileCheck:
        return CSVFileCheck(self.file_path)

    def _parse(self) -> DataFrame | None:
        # Every row is parsed, because a malformed row can be anywhere in the file.
        try:
            return read_csv(self.file_path, header=0)
        except errors.EmptyDataError:
            # Only blank lines: the file is empty, not corrupted, so not_empty() reports it.
            return None
        except UnicodeDecodeError as e:
            raise CorruptedFileError(f"File '{self.file_path}' has a spoofed extension or an encoding other than UTF-8.") from e
        except errors.ParserError as e:
            raise CorruptedFileError(f"File '{self.file_path}' is malformed: {e}") from e


class ExcelReader(Reader):
    """Reader for Excel files (``.xlsx`` and ``.xls``). Only the first worksheet is read."""

    def _get_file_check(self) -> ExcelFileCheck:
        return ExcelFileCheck(self.file_path)

    def _parse(self) -> DataFrame:
        # The whole worksheet is parsed, because the damage can be past the first row.
        try:
            return read_excel(self.file_path, header=0)
        except (KeyError, ValueError, XLRDError) as e:
            # pandas raises ValueError both for encrypted workbooks and for files that are not Excel at all.
            raise CorruptedFileError(f"File '{self.file_path}' is encrypted or is not an Excel file.") from e
        except BadZipFile as e:
            raise CorruptedFileError(f"File '{self.file_path}' has a spoofed extension or is incomplete.") from e
        except (ParseError, InvalidFileException) as e:
            raise CorruptedFileError(f"File '{self.file_path}' is corrupted.") from e


class SQLiteReader(Reader):
    """Reader for SQLite database files with a single table."""

    def _get_file_check(self) -> SQLiteFileCheck:
        return SQLiteFileCheck(self.file_path)

    def _parse(self) -> DataFrame:
        # One connection for the whole read, so the database cannot change between the checks and the read.
        conexion = sql.connect(self.file_path)
        try:
            self._check_integrity(conexion)
            tables = [name for (name,) in conexion.execute(f"SELECT name {_SQLITE_TABLES}")]
            self._get_file_check().single_table(tables)
            # Quoted so names with spaces or reserved words work; an inner quote is escaped by doubling it.
            quoted_name = '"' + tables[0].replace('"', '""') + '"'
            return read_sql_query(f"SELECT * FROM {quoted_name}", conexion)
        finally:
            conexion.close()

    def _check_integrity(self, conexion: sql.Connection) -> None:
        """Raise ``CorruptedFileError`` if the database is damaged or the file is not a SQLite database."""
        try:
            # A file that is not a database, or a badly damaged one, makes the pragma itself raise instead of reporting.
            result = conexion.execute("PRAGMA integrity_check").fetchone()[0]
        except sql.DatabaseError as e:
            raise CorruptedFileError(f"File '{self.file_path}' is corrupted or is not a SQLite database.") from e
        if result != "ok":
            raise CorruptedFileError(f"File '{self.file_path}' is corrupted: {result}")


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
    raise UnsupportedExtensionError(f"Extension '{ext}' is not supported.")
