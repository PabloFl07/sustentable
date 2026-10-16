import sqlite3
import zipfile
from pathlib import Path

import pytest

from sustentable.data_import import (
    _READERS,
    CSVFileCheck,
    CSVReader,
    ExcelFileCheck,
    ExcelReader,
    FileCheck,
    Reader,
    SQLiteFileCheck,
    SQLiteReader,
    get_reader,
)

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def example_empty_csv(tmp_path):
    path = Path(tmp_path / "example.csv")
    path.touch()
    return path


@pytest.fixture
def example_csv(tmp_path):
    path = Path(tmp_path / "example.csv")
    path.write_text("name,age\npablo,19\ndiego,21")
    return path


@pytest.fixture
def example_empty_xlsx(tmp_path):
    path = Path(tmp_path / "example.xlsx")
    path.touch()
    return path


# Minimal parts of a valid .xlsx (a zip of XML files), so no Excel library is needed.
_XLSX_PARTS = {
    "[Content_Types].xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml"'
        ' ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    ),
    "_rels/.rels": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
        ' Target="xl/workbook.xml"/>'
        "</Relationships>"
    ),
    "xl/workbook.xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    ),
    "xl/_rels/workbook.xml.rels": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"'
        ' Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    ),
    "xl/worksheets/sheet1.xml": (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        '<row r="1"><c r="A1" t="inlineStr"><is><t>name</t></is></c><c r="B1" t="inlineStr"><is><t>age</t></is></c></row>'
        '<row r="2"><c r="A2" t="inlineStr"><is><t>pablo</t></is></c><c r="B2"><v>19</v></c></row>'
        '<row r="3"><c r="A3" t="inlineStr"><is><t>diego</t></is></c><c r="B3"><v>21</v></c></row>'
        "</sheetData></worksheet>"
    ),
}


@pytest.fixture
def example_xlsx(tmp_path):
    path = Path(tmp_path / "example.xlsx")
    with zipfile.ZipFile(path, "w") as xlsx:
        for name, content in _XLSX_PARTS.items():
            xlsx.writestr(name, content)
    return path


@pytest.fixture
def example_empty_sqlite(tmp_path):
    path = Path(tmp_path / "example.db")
    path.touch()
    return path


@pytest.fixture
def example_sqlite(tmp_path):
    path = Path(tmp_path / "example.db")
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("CREATE TABLE people (name TEXT, age INTEGER)")
        connection.executemany("INSERT INTO people VALUES (?, ?)", [("pablo", 19), ("diego", 21)])
    connection.close()
    return path


# ----------------------------------------------------------------------------
# exists(): the path points to something
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_check_class", "fixture_name"),
    [
        (CSVFileCheck, "example_csv"),
        (ExcelFileCheck, "example_xlsx"),
        (SQLiteFileCheck, "example_sqlite"),
    ],
)
def test_file_exists(request, file_check_class, fixture_name):
    """An existing file of each format passes ``exists()``."""
    # Fixtures cannot be passed to parametrize directly, so their names are resolved at run time.
    file_check_class(request.getfixturevalue(fixture_name)).exists()


def test_not_file_exists(tmp_path):
    """A path with nothing behind it fails ``exists()``."""
    with pytest.raises(FileNotFoundError):
        CSVFileCheck(tmp_path / "imaginary.csv").exists()


# ----------------------------------------------------------------------------
# is_file(): the path is a regular file
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_check_class", "fixture_name"),
    [
        (CSVFileCheck, "example_csv"),
        (ExcelFileCheck, "example_xlsx"),
        (SQLiteFileCheck, "example_sqlite"),
    ],
)
def test_is_file(request, file_check_class, fixture_name):
    """A regular file of each format passes ``is_file()``."""
    file_check_class(request.getfixturevalue(fixture_name)).is_file()


def test_symlink_to_file_is_file(tmp_path, example_csv):
    """A symbolic link to a regular file passes ``is_file()``, because the link is followed."""
    link = tmp_path / "link.csv"
    link.symlink_to(example_csv)
    CSVFileCheck(link).is_file()


@pytest.mark.parametrize(
    ("file_check_class", "folder_name"),
    [
        (CSVFileCheck, "data.csv"),
        (ExcelFileCheck, "data.xlsx"),
        (SQLiteFileCheck, "data.db"),
    ],
)
def test_directory_is_not_file(tmp_path, file_check_class, folder_name):
    """A directory named with a supported extension fails ``is_file()``."""
    # The directory exists, so exists() alone would let it through.
    folder = tmp_path / folder_name
    folder.mkdir()
    with pytest.raises(ValueError, match="is not a file"):
        file_check_class(folder).is_file()


def test_validate_rejects_directory(tmp_path):
    """``validate()`` rejects a directory before reaching the content checks."""
    # If is_file() were skipped, validate() would end in NotImplementedError instead of this ValueError.
    folder = tmp_path / "data.csv"
    folder.mkdir()
    with pytest.raises(ValueError, match="is not a file"):
        CSVReader(folder).validate()


# ----------------------------------------------------------------------------
# supported_extension(): only the extensions of each format are accepted
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_check_class", "file_name"),
    [
        (CSVFileCheck, "data.csv"),
        (CSVFileCheck, "DATA.CSV"),
        (ExcelFileCheck, "data.xlsx"),
        (ExcelFileCheck, "data.xls"),
        (ExcelFileCheck, "DATA.XLSX"),
        (SQLiteFileCheck, "data.db"),
        (SQLiteFileCheck, "data.sqlite"),
        (SQLiteFileCheck, "data.sqlite3"),
        (SQLiteFileCheck, "DATA.DB"),
    ],
)
def test_supported_extension(file_check_class, file_name):
    """Every extension of a format is accepted by its check, regardless of case."""
    # The extension check only looks at the name, so the files do not need to exist.
    file_check_class(Path(file_name)).supported_extension()


@pytest.mark.parametrize(
    ("file_check_class", "file_name"),
    [
        (CSVFileCheck, "data.txt"),
        (CSVFileCheck, "data.xlsx"),
        (CSVFileCheck, "data"),
        (ExcelFileCheck, "data.csv"),
        (ExcelFileCheck, "data.xlm"),
        (SQLiteFileCheck, "data.csv"),
        (SQLiteFileCheck, "data.sql"),
    ],
)
def test_unsupported_extension(file_check_class, file_name):
    """Extensions of other formats, unknown extensions and missing extensions are rejected."""
    with pytest.raises(ValueError):
        file_check_class(Path(file_name)).supported_extension()


@pytest.mark.parametrize(
    ("reader_class", "file_name"),
    [
        (CSVReader, "data.xlsx"),
        (CSVReader, "data.db"),
        (ExcelReader, "data.csv"),
        (ExcelReader, "data.sqlite"),
        (SQLiteReader, "data.csv"),
        (SQLiteReader, "data.xls"),
    ],
)
def test_wrong_reader_for_extension(tmp_path, reader_class, file_name):
    """A reader rejects a file of another format, even though the module supports that extension."""
    # The file has content so validate() gets past exists(), is_file() and reaches the extension check.
    path = tmp_path / file_name
    path.write_text("some data")
    with pytest.raises(ValueError, match="not supported"):
        reader_class(path).validate()


# ----------------------------------------------------------------------------
# not_empty(): files without data are rejected
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_check_class", "fixture_name"),
    [
        (CSVFileCheck, "example_empty_csv"),
        (ExcelFileCheck, "example_empty_xlsx"),
        (SQLiteFileCheck, "example_empty_sqlite"),
    ],
)
def test_empty_file(request, file_check_class, fixture_name):
    """A zero byte file fails ``not_empty()`` for every format."""
    # The size check runs first and short-circuits, so the unimplemented _not_empty() is never called.
    with pytest.raises(ValueError):
        file_check_class(request.getfixturevalue(fixture_name)).not_empty()


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="CSVFileCheck._not_empty() is not implemented yet")
def test_not_empty_csv(example_csv):
    """A CSV with rows passes ``not_empty()``."""
    # strict=True turns an unexpected pass into a failure, so the marker must be removed once _not_empty() exists.
    CSVFileCheck(example_csv).not_empty()


# ----------------------------------------------------------------------------
# Reader.file_path: always stored as a Path
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("file_path", ["data.csv", Path("data.csv")])
def test_file_path_stored_as_path(file_path):
    """The path given to the constructor is stored as a ``Path``, whether it was a ``str`` or a ``Path``."""
    reader = CSVReader(file_path)
    assert isinstance(reader.file_path, Path)
    assert reader.file_path == Path("data.csv")


def test_file_path_setter_converts_to_path():
    """Assigning a ``str`` to ``file_path`` after construction also stores a ``Path``."""
    reader = CSVReader("data.csv")
    reader.file_path = "other.csv"
    assert isinstance(reader.file_path, Path)
    assert reader.file_path == Path("other.csv")


@pytest.mark.parametrize("file_path", [None, 123, ["data.csv"]])
def test_file_path_invalid_type(file_path):
    """A path that is neither a ``str`` nor a ``Path`` raises ``TypeError``."""
    with pytest.raises(TypeError):
        CSVReader(file_path)


# ----------------------------------------------------------------------------
# get_reader(): picks the reader from the extension
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_path", "reader_class"),
    [
        ("data.csv", CSVReader),
        ("data.xlsx", ExcelReader),
        ("data.xls", ExcelReader),
        ("data.db", SQLiteReader),
        ("data.sqlite", SQLiteReader),
        ("data.sqlite3", SQLiteReader),
        ("DATA.CSV", CSVReader),
        (Path("data.XLSX"), ExcelReader),
    ],
)
def test_get_reader(file_path, reader_class):
    """``get_reader()`` returns the reader for the extension, ignoring case, and keeps the given path."""
    reader = get_reader(file_path)
    assert isinstance(reader, reader_class)
    assert reader.file_path == Path(file_path)


def test_get_reader_covers_every_registered_extension():
    """Every extension registered in ``_READERS`` leads ``get_reader()`` to its reader."""
    # Reads the registry instead of a fixed list, so new formats are covered without editing this test.
    for extensions, reader_class in _READERS:
        for ext in extensions:
            assert isinstance(get_reader(f"data{ext}"), reader_class)


@pytest.mark.parametrize("file_path", ["data.txt", "data.json", "data"])
def test_get_reader_unsupported_extension(file_path):
    """``get_reader()`` raises ``ValueError`` for an unknown or missing extension."""
    with pytest.raises(ValueError):
        get_reader(file_path)


# ----------------------------------------------------------------------------
# _get_file_check() and file_check: each reader builds its own check
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reader_class", "file_check_class"),
    [
        (CSVReader, CSVFileCheck),
        (ExcelReader, ExcelFileCheck),
        (SQLiteReader, SQLiteFileCheck),
    ],
)
def test_get_file_check(reader_class, file_check_class):
    """Each reader builds the check for its own format, pointing to the same path."""
    # The extension is irrelevant here: no check is run, only the type of the check is inspected.
    reader: Reader = reader_class("data.file")
    file_check = reader._get_file_check()
    assert isinstance(file_check, FileCheck)
    assert isinstance(file_check, file_check_class)
    assert file_check.file_path == reader.file_path


def test_file_check_follows_file_path_changes():
    """After changing ``file_path``, ``file_check`` points to the new path instead of the old one."""
    reader = CSVReader("first.csv")
    reader.file_path = "second.csv"
    assert reader.file_check.file_path == Path("second.csv")


# ----------------------------------------------------------------------------
# validate(): a valid file passes every check
# ----------------------------------------------------------------------------


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Content and corruption checks are not implemented yet")
@pytest.mark.parametrize(
    ("reader_class", "fixture_name"),
    [
        (CSVReader, "example_csv"),
        (ExcelReader, "example_xlsx"),
        (SQLiteReader, "example_sqlite"),
    ],
)
def test_valid_file_passes_validate(request, reader_class, fixture_name):
    """A valid file of each format passes every check in ``validate()``."""
    reader_class(request.getfixturevalue(fixture_name)).validate()
