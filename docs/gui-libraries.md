# GUI libraries for Sustentable

This note compares Pysual 0.1.2, Tkinter (CPython 3.14), and PySide6 6.12 for the Sustentable interface. One team member maintains [Pysual](https://github.com/PabloMP2P/Pysual). Flexibility is weighted toward Pysual, because a missing control can be added in that library; the other criteria describe each library as it is today. wxPython, Dear PyGui, and Kivy were not compared.

## Comparison

| Criterion | Pysual 0.1.2 | Tkinter | PySide6 6.12 |
| --- | --- | --- | --- |
| Ease of use | Small `App` API (`build()`, `name_on_event`). Alpha, so the API can change. Short curve for this team because the maintainer is on it. | Standard library and short examples. A table is `ttk.Treeview` plus scrollbars, and layouts get awkward. | Full Qt widget set. Layouts, models, and signals take longer to learn. |
| Windows, macOS, Linux | Native window, terminal, or web. Window wheels for Windows x64, Linux x64, and Apple Silicon; other machines use the pure-Python terminal and web backends. | Included with official Windows and macOS installers. Some Linux distributions ship Tk as a separate system package. | Qt widgets and wheels for the common desktop architectures. Python 3.14 is supported since 6.10.1; 6.12 requires `>=3.10,<3.16`. |
| License | MIT. The project can use and modify it. | Python Software Foundation License, as part of CPython. | LGPL-3.0-only, or GPL-2.0-only, or GPL-3.0-only. Distribution must meet the LGPL obligations for Qt. |
| Documentation and community | API guide and examples on GitHub. Alpha, small user community; questions can go to the maintainer on the team. | Python documentation and many years of examples. | Qt for Python docs and the Qt reference. Large community. |
| Components | Window, button, label, multiline `TextBox`, single-choice `Dropdown`, `CheckBox`, numeric field, scrolling `DataGrid` (20,000 rows, 64 columns, 200,000 cells), alert, and image. `open_text_file()` opens the OS dialog and then reads the file as UTF-8 (8 MiB max), with no extension filter, so Excel and SQLite need a new path-only call. No Matplotlib canvas: a figure can be shown through `Image`, or a custom `Control` can paint it. `LineChart` only covers simple series. | `filedialog.askopenfilename(filetypes=...)`, `Text`, `ttk.Combobox`, extended `Listbox`, `Treeview`, and `FigureCanvasTkAgg`. | `QFileDialog` with name filters, `QTableView`, `QComboBox`, extended `QListWidget`, `QPlainTextEdit`, and `FigureCanvasQTAgg`. |
| `uv add` | `uv add pysual`. On PyPI, Python >=3.11, including 3.14. | No package to add. On Linux the system Tk package can still be missing. | `uv add pyside6` works. The install is large because it brings Qt. |

## Decision

The team will build the interface with Pysual. Tkinter and PySide6 already have a filtered file dialog and a Matplotlib canvas; Pysual 0.1.2 does not. Both gaps can be closed inside Pysual, because the OS dialogs already exist and custom controls are supported. One MIT library also avoids PySide6's LGPL obligations and Tkinter's older table widgets. Pysual is alpha, its community is small, and native window wheels do not cover every architecture; that is acceptable because a maintainer is on the team and the application only needs a desktop window.

The project will add it with `uv add pysual`. The other two libraries stay out of `pyproject.toml`. The proof of concept goes under `prototipos/` and is not imported by `src/sustentable/`. It will exercise a titled window, a file button and the chosen path, a scrolling sample table, a dropdown, a text field, and a button that shows the selected or typed value. Notes on what broke will be added here after that window runs.