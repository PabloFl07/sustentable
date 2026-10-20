"""Example files for the data import tests.

Every fixture writes one file into ``tmp_path`` and returns its path. The valid ones hold the same two rows,
(José, 19) and (Begoña, 21), so the tests can check what ``read()`` returns.
"""

import sqlite3
import zipfile
from pathlib import Path

import pytest
from openpyxl import Workbook

CSV_TEXT = "name,age\nJosé,19\nBegoña,21\n"


def _write_xlsx(path: Path, *sheets: list[tuple]) -> Path:
    """Write an .xlsx with one worksheet per item of ``sheets``, each one a list of rows."""
    workbook = Workbook()
    workbook.remove(workbook.active)
    for number, rows in enumerate(sheets, start=1):
        sheet = workbook.create_sheet(f"Sheet{number}")
        for row in rows:
            sheet.append(row)
    workbook.save(path)
    return path


def _write_sqlite(path: Path, *statements: str) -> Path:
    """Create a SQLite database and run ``statements`` on it."""
    connection = sqlite3.connect(path)
    with connection:
        for statement in statements:
            connection.execute(statement)
    connection.close()
    return path


def _truncate(path: Path) -> Path:
    """Cut the file to half its size, as an interrupted download or copy would."""
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    return path


# ----------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------


@pytest.fixture
def example_csv(tmp_path):
    """Valid CSV in UTF-8."""
    path = tmp_path / "example.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def example_uppercase_csv(tmp_path):
    """Valid CSV with the extension in uppercase."""
    path = tmp_path / "EXAMPLE.CSV"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def example_bom_csv(tmp_path):
    """Valid CSV in UTF-8 with BOM, as Excel saves it."""
    path = tmp_path / "example.csv"
    path.write_text(CSV_TEXT, encoding="utf-8-sig")
    return path


@pytest.fixture
def example_semicolon_csv(tmp_path):
    """Valid CSV separated by ';', as Excel in Spanish saves it."""
    path = tmp_path / "example.csv"
    path.write_text("name;age\nJosé;19\nBegoña;21\n", encoding="utf-8")
    return path


@pytest.fixture
def example_empty_csv(tmp_path):
    """CSV of zero bytes."""
    path = tmp_path / "example.csv"
    path.touch()
    return path


@pytest.fixture
def example_header_only_csv(tmp_path):
    """CSV with the header and no rows."""
    path = tmp_path / "example.csv"
    path.write_text("name,age\n", encoding="utf-8")
    return path


@pytest.fixture
def example_blank_lines_csv(tmp_path):
    """CSV with only spaces, tabs and line breaks."""
    path = tmp_path / "example.csv"
    path.write_text("   \n\t\n", encoding="utf-8")
    return path


@pytest.fixture
def example_binary_csv(tmp_path):
    """Binary file (the start of a PNG) with a .csv extension."""
    path = tmp_path / "example.csv"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe" * 20)
    return path


@pytest.fixture
def example_latin1_csv(tmp_path):
    """CSV in Latin-1, which is rejected because only UTF-8 is accepted."""
    path = tmp_path / "example.csv"
    path.write_text(CSV_TEXT, encoding="latin-1")
    return path


@pytest.fixture
def example_ragged_csv(tmp_path):
    """CSV with a row that has more fields than the header (treated as corrupted by design)."""
    path = tmp_path / "example.csv"
    path.write_text("name,age\nJosé,19\nBegoña,21,x,y\n", encoding="utf-8")
    return path


@pytest.fixture
def example_unclosed_quote_csv(tmp_path):
    """CSV with a quote that is never closed."""
    path = tmp_path / "example.csv"
    path.write_text('name,age\n"José,19\n', encoding="utf-8")
    return path


@pytest.fixture
def example_unreadable_csv(tmp_path):
    """Valid CSV without read permissions."""
    path = tmp_path / "example.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    path.chmod(0)
    return path


# ----------------------------------------------------------------------------
# Excel
# ----------------------------------------------------------------------------


@pytest.fixture
def example_xlsx(tmp_path):
    """Valid .xlsx, with age stored as a number."""
    return _write_xlsx(tmp_path / "example.xlsx", [("name", "age"), ("José", 19), ("Begoña", 21)])


@pytest.fixture
def example_empty_xlsx(tmp_path):
    """.xlsx of zero bytes."""
    path = tmp_path / "example.xlsx"
    path.touch()
    return path


@pytest.fixture
def example_header_only_xlsx(tmp_path):
    """.xlsx with the header and no rows."""
    return _write_xlsx(tmp_path / "example.xlsx", [("name", "age")])


@pytest.fixture
def example_no_rows_xlsx(tmp_path):
    """.xlsx with one blank worksheet."""
    return _write_xlsx(tmp_path / "example.xlsx", [])


@pytest.fixture
def example_text_xlsx(tmp_path):
    """CSV text with a .xlsx extension."""
    path = tmp_path / "example.xlsx"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def example_truncated_xlsx(example_xlsx):
    """.xlsx cut in half."""
    return _truncate(example_xlsx)


@pytest.fixture
def example_broken_xml_xlsx(example_xlsx):
    """.xlsx with a valid zip whose worksheet XML has unclosed tags."""
    with zipfile.ZipFile(example_xlsx) as xlsx:
        parts = {name: xlsx.read(name) for name in xlsx.namelist()}
    parts["xl/worksheets/sheet1.xml"] = b"<worksheet><sheetData><row"
    with zipfile.ZipFile(example_xlsx, "w") as xlsx:
        for name, content in parts.items():
            xlsx.writestr(name, content)
    return example_xlsx


@pytest.fixture
def example_two_sheets_xlsx(tmp_path):
    """.xlsx with two worksheets with data."""
    return _write_xlsx(tmp_path / "example.xlsx", [("name", "age"), ("José", 19)], [("city",), ("Lugo",)])


@pytest.fixture
def example_unreadable_xlsx(example_xlsx):
    """Valid .xlsx without read permissions."""
    example_xlsx.chmod(0)
    return example_xlsx


# ----------------------------------------------------------------------------
# SQLite
# ----------------------------------------------------------------------------


@pytest.fixture
def example_sqlite(tmp_path):
    """Valid database with one table."""
    return _write_sqlite(
        tmp_path / "example.db",
        "CREATE TABLE people (name TEXT, age INTEGER)",
        "INSERT INTO people VALUES ('José', 19), ('Begoña', 21)",
    )


@pytest.fixture
def example_autoincrement_sqlite(tmp_path):
    """Valid database with one table plus the internal sqlite_sequence table, a view and an index."""
    return _write_sqlite(
        tmp_path / "example.db",
        "CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER)",
        "INSERT INTO people (name, age) VALUES ('José', 19), ('Begoña', 21)",
        "CREATE VIEW adults AS SELECT name FROM people WHERE age >= 18",
        "CREATE INDEX people_name ON people (name)",
    )


@pytest.fixture
def example_empty_sqlite(tmp_path):
    """Database file of zero bytes."""
    path = tmp_path / "example.db"
    path.touch()
    return path


@pytest.fixture
def example_no_tables_sqlite(tmp_path):
    """Valid database without tables."""
    return _write_sqlite(tmp_path / "example.db", "PRAGMA user_version = 1")


@pytest.fixture
def example_empty_table_sqlite(tmp_path):
    """Database with one table and no rows."""
    return _write_sqlite(tmp_path / "example.db", "CREATE TABLE people (name TEXT, age INTEGER)")


@pytest.fixture
def example_text_sqlite(tmp_path):
    """CSV text with a .db extension."""
    path = tmp_path / "example.db"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def example_truncated_sqlite(tmp_path):
    """Database cut in half. It has enough rows to span many pages, so the cut always loses table pages."""
    path = _write_sqlite(
        tmp_path / "example.db",
        "CREATE TABLE people (name TEXT, age INTEGER)",
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 2000)"
        " INSERT INTO people SELECT printf('person %050d', i), i FROM n",
    )
    return _truncate(path)


@pytest.fixture
def example_two_tables_sqlite(tmp_path):
    """Database with two tables."""
    return _write_sqlite(
        tmp_path / "example.db",
        "CREATE TABLE people (name TEXT, age INTEGER)",
        "INSERT INTO people VALUES ('José', 19), ('Begoña', 21)",
        "CREATE TABLE places (name TEXT)",
    )


@pytest.fixture
def example_unreadable_sqlite(example_sqlite):
    """Valid database without read permissions."""
    example_sqlite.chmod(0)
    return example_sqlite


# ----------------------------------------------------------------------------
# Paths that are not a valid file
# ----------------------------------------------------------------------------


@pytest.fixture
def example_missing_csv(tmp_path):
    """Path to a .csv that does not exist."""
    return tmp_path / "missing.csv"


@pytest.fixture
def example_directory_csv(tmp_path):
    """Directory named like a CSV file."""
    path = tmp_path / "example.csv"
    path.mkdir()
    return path


@pytest.fixture
def example_txt(tmp_path):
    """Valid CSV content with a .txt extension."""
    path = tmp_path / "example.txt"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def example_no_extension(tmp_path):
    """Valid CSV content without an extension."""
    path = tmp_path / "example"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path
