"""Tests for the data import module.

The example files are fixtures defined in ``conftest.py``. Fixtures cannot be passed to ``parametrize`` directly,
so the tests receive their names and get the file with ``request.getfixturevalue()``.
"""

import os
from pathlib import Path

import pytest
from pandas import DataFrame
from pandas.api.types import is_integer_dtype

from sustentable.data_import import _READERS, CSVReader, ExcelReader, SQLiteReader, get_reader
from sustentable.exceptions import (
    CorruptedFileError,
    DataImportError,
    EmptyFileError,
    MissingFileError,
    MultipleTablesError,
    NotAFileError,
    UnreadableFileError,
    UnsupportedExtensionError,
)


def _assert_people(data: DataFrame) -> None:
    """Check that ``data`` holds the two rows of the valid example files, with ``age`` as integers."""
    assert isinstance(data, DataFrame)
    assert list(data.columns) == ["name", "age"]
    assert data.values.tolist() == [["José", 19], ["Begoña", 21]]
    # The xlsx stores age as a number and the CSV as text, so both must end up as integers.
    assert is_integer_dtype(data["age"])


# ----------------------------------------------------------------------------
# read(): a valid file returns its data
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name",
    [
        "example_csv",
        "example_uppercase_csv",
        "example_bom_csv",
        pytest.param(
            "example_semicolon_csv",
            marks=pytest.mark.xfail(strict=True, reason="CSVReader does not detect the separator, so it reads a single 'name;age' column"),
        ),
        "example_xlsx",
        "example_sqlite",
    ],
)
def test_read_returns_file_content(request, fixture_name):
    """``read()`` returns the columns and rows of the file, with the accents and ñ of its content."""
    path = request.getfixturevalue(fixture_name)
    _assert_people(get_reader(path).read())


@pytest.mark.parametrize("absolute", [True, False], ids=["absolute", "relative"])
@pytest.mark.parametrize("fixture_name", ["example_csv", "example_xlsx", "example_sqlite"])
def test_read_with_path_formats(request, monkeypatch, tmp_path, fixture_name, absolute):
    """``get_reader()`` and ``read()`` accept absolute and relative paths with spaces, accents and ñ."""
    source = request.getfixturevalue(fixture_name)
    relative = Path("carpeta con espacios") / f"canción del año{source.suffix}"
    (tmp_path / relative).parent.mkdir()
    (tmp_path / relative).write_bytes(source.read_bytes())

    # A relative path is resolved against the working directory, so it is moved to tmp_path.
    monkeypatch.chdir(tmp_path)
    file_path = str(tmp_path / relative) if absolute else str(relative)

    reader = get_reader(file_path)
    assert reader.file_path == Path(file_path)
    _assert_people(reader.read())


def test_read_follows_symlink(tmp_path, example_csv):
    """A symbolic link to a valid file is read like the file itself."""
    link = tmp_path / "link.csv"
    # Windows only lets administrators or Developer Mode create symlinks; anywhere else this raises OSError.
    try:
        link.symlink_to(example_csv)
    except OSError as error:
        pytest.skip(f"Symbolic links cannot be created here: {error}")
    _assert_people(CSVReader(link).read())


def test_read_sqlite_ignores_internal_tables_views_and_indexes(example_autoincrement_sqlite):
    """``sqlite_sequence``, a view and an index do not count as extra tables."""
    assert len(SQLiteReader(example_autoincrement_sqlite).read()) == 2


# ----------------------------------------------------------------------------
# read(): each invalid file raises its own exception
# ----------------------------------------------------------------------------

_NO_PERMISSIONS = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="Needs POSIX permissions and a user other than root"
)


@pytest.mark.parametrize(
    ("reader_class", "fixture_name", "error"),
    [
        # The path does not lead to a regular file.
        (CSVReader, "example_missing_csv", MissingFileError),
        (CSVReader, "example_directory_csv", NotAFileError),
        # The file exists but the user cannot read it. Skipped where chmod(0) does not stop reading: root or Windows.
        pytest.param(CSVReader, "example_unreadable_csv", UnreadableFileError, marks=_NO_PERMISSIONS),
        pytest.param(ExcelReader, "example_unreadable_xlsx", UnreadableFileError, marks=_NO_PERMISSIONS),
        pytest.param(SQLiteReader, "example_unreadable_sqlite", UnreadableFileError, marks=_NO_PERMISSIONS),
        # The extension is unknown, missing, or belongs to another reader.
        (CSVReader, "example_txt", UnsupportedExtensionError),
        (CSVReader, "example_no_extension", UnsupportedExtensionError),
        (CSVReader, "example_xlsx", UnsupportedExtensionError),
        (ExcelReader, "example_csv", UnsupportedExtensionError),
        (SQLiteReader, "example_xlsx", UnsupportedExtensionError),
        # The file can be read but has no data rows.
        (CSVReader, "example_empty_csv", EmptyFileError),
        (CSVReader, "example_header_only_csv", EmptyFileError),
        (CSVReader, "example_blank_lines_csv", EmptyFileError),
        (ExcelReader, "example_empty_xlsx", EmptyFileError),
        (ExcelReader, "example_header_only_xlsx", EmptyFileError),
        (ExcelReader, "example_no_rows_xlsx", EmptyFileError),
        (SQLiteReader, "example_empty_sqlite", EmptyFileError),
        (SQLiteReader, "example_no_tables_sqlite", EmptyFileError),
        (SQLiteReader, "example_empty_table_sqlite", EmptyFileError),
        # The file cannot be parsed.
        (CSVReader, "example_binary_csv", CorruptedFileError),
        (CSVReader, "example_latin1_csv", CorruptedFileError),
        (CSVReader, "example_ragged_csv", CorruptedFileError),
        (CSVReader, "example_unclosed_quote_csv", CorruptedFileError),
        (ExcelReader, "example_text_xlsx", CorruptedFileError),
        (ExcelReader, "example_truncated_xlsx", CorruptedFileError),
        (ExcelReader, "example_broken_xml_xlsx", CorruptedFileError),
        (SQLiteReader, "example_text_sqlite", CorruptedFileError),
        (SQLiteReader, "example_truncated_sqlite", CorruptedFileError),
        # There is more than one table, so there is no way to know which one to load.
        (SQLiteReader, "example_two_tables_sqlite", MultipleTablesError),
    ],
)
# Test names such as "CSVReader-example_empty_csv-EmptyFileError".
def test_read_raises_specific_error(request, reader_class, fixture_name, error):
    """``read()`` raises the exception of the problem, not a library exception or the one of another problem.

    ``read()`` instead of ``validate()`` because the content is checked while parsing, so ``validate()`` alone only
    catches the file system problems.
    """
    path = request.getfixturevalue(fixture_name)
    with pytest.raises(error):
        reader_class(path).read()


# ----------------------------------------------------------------------------
# get_reader(): picks the reader from the extension
# ----------------------------------------------------------------------------


def test_get_reader_covers_every_registered_extension():
    """Every extension registered in ``_READERS`` leads ``get_reader()`` to its reader, in lowercase and uppercase."""
    # Reads the registry instead of a fixed list, so new formats are covered without editing this test.
    for extensions, reader_class in _READERS:
        for ext in extensions:
            assert isinstance(get_reader(f"data{ext}"), reader_class)
            assert isinstance(get_reader(Path(f"DATA{ext.upper()}")), reader_class)


@pytest.mark.parametrize("file_path", ["data.txt", "data.json", "data"])
def test_get_reader_unsupported_extension(file_path):
    """``get_reader()`` rejects an unknown or missing extension before any file is opened."""
    with pytest.raises(UnsupportedExtensionError):
        get_reader(file_path)


# ----------------------------------------------------------------------------
# Reader.file_path: always a Path, and the checks follow it
# ----------------------------------------------------------------------------


def test_file_check_follows_file_path_changes():
    """Assigning a ``str`` to ``file_path`` stores a ``Path``, and ``file_check`` points to the new path instead of the old one."""
    reader = CSVReader("first.csv")
    reader.file_path = "second.csv"
    assert reader.file_path == Path("second.csv")
    assert isinstance(reader.file_path, Path)
    assert reader.file_check.file_path == Path("second.csv")


@pytest.mark.parametrize("file_path", [None, 123, ["data.csv"]])
def test_file_path_invalid_type(file_path):
    """A path that is neither a ``str`` nor a ``Path`` raises ``TypeError``."""
    with pytest.raises(TypeError):
        CSVReader(file_path)


# ----------------------------------------------------------------------------
# Exceptions: one base class for the interface, builtin parents for old callers
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_class", "builtin_class"),
    [
        (MissingFileError, FileNotFoundError),
        (NotAFileError, ValueError),
        (UnreadableFileError, PermissionError),
        (UnsupportedExtensionError, ValueError),
        (CorruptedFileError, ValueError),
        (EmptyFileError, ValueError),
        (MultipleTablesError, ValueError),
    ],
)
def test_exception_hierarchy(error_class, builtin_class):
    """Every error derives from ``DataImportError`` and from the builtin exception the module raised before."""
    assert issubclass(error_class, DataImportError)
    assert issubclass(error_class, builtin_class)
