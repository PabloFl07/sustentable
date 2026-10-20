"""Exceptions raised while importing a data file.

Every exception derives from ``DataImportError``, so the interface can catch all of them with a single ``except`` and
show the message, which is always written for the user and names the file. Each one also derives from the builtin
exception the module raised before, so code that catches ``ValueError`` or ``FileNotFoundError`` keeps working.
"""


class DataImportError(Exception):
    """Base class for every error found while checking or reading a data file."""


class MissingFileError(DataImportError, FileNotFoundError):
    """The path does not point to anything."""


class NotAFileError(DataImportError, ValueError):
    """The path exists but is not a regular file, such as a directory with a supported extension."""


class UnreadableFileError(DataImportError, PermissionError):
    """The file exists but the user has no permission to read it."""


class UnsupportedExtensionError(DataImportError, ValueError):
    """The extension of the file is not one the reader, or the module, can read."""


class CorruptedFileError(DataImportError, ValueError):
    """The file cannot be parsed: it is damaged, incomplete, encrypted or not really of the format its extension says."""


class EmptyFileError(DataImportError, ValueError):
    """The file can be parsed but contains no data rows."""


class MultipleTablesError(DataImportError, ValueError):
    """The SQLite database has more than one table, so there is no way to know which one to load."""
