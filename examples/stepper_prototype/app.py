"""Stepper UX playground. Run with ``uv run python -m examples.stepper_prototype.app``."""

from pysual import App, Container, Dropdown, Label, UiEvent, autoconfig, get_theme  # noqa: F401 (autoconfig applies environment defaults)

from sustentable.gui.controls.sidebar_button import SidebarButton, with_sidebar_button

from .workflow import Workflow

OPTIONS = (
    ("Layout", ("Vertical", "Horizontal", "Vertical rail")),
    ("Steps", ("3 · Separate steps", "2 · Data + columns", "6 · With placeholders")),
    ("Navigation", ("Free", "Guided")),
    ("Validation", ("On continue", "Live hints", "None")),
    ("Data grid", ("Shared preview", "Inside each step")),
)


class StepperWindow(App):
    """Let reviewers try alternatives without changing code or loading files."""

    def build(self) -> None:
        """Keep the same app theme and expose five independent design choices."""
        self.title, self.width, self.height = "Sustentable · Stepper alternatives", 1120, 680
        self.layout, self.padding, self.spacing = "stack", 16, 12
        self.theme = with_sidebar_button(get_theme("modern_dark"))
        header = Container(parent=self, layout="stack", direction="horizontal", spacing=8, height=32)
        Label(parent=header, text="Stepper alternatives", font_size=20, flex=1)
        reset = SidebarButton(parent=header, text="Reset example", width=144)
        reset.click.connect(self._rebuild)
        row = Container(parent=self, layout="stack", direction="horizontal", spacing=16, height=64)
        self.choices = []
        for title, items in OPTIONS:
            cell = Container(parent=row, layout="stack", spacing=4, flex=1)
            Label(parent=cell, text=title, height=24)
            choice = Dropdown(parent=cell, items=items, selected_index=0, height=36, font_size=14)
            choice.changed.connect(self._rebuild)
            self.choices.append(choice)
        Label(
            parent=self,
            text="UI only · Sample data. Changing settings resets the example. Guided navigation checks earlier steps.",
            height=24,
            font_size=14,
        )
        self.stage = Container(parent=self, layout="stack", flex=1)
        self.demo: Workflow | None = None
        self._rebuild()

    def _rebuild(self, event: UiEvent | None = None) -> None:
        if not hasattr(self, "stage"):
            return
        if self.demo is not None:
            self.demo.destroy()
        alignment, count, navigation, validation, grid = self.choices
        self.demo = Workflow(
            parent=self.stage,
            flex=1,
            orientation=("vertical", "horizontal", "rail")[alignment.selected_index],
            steps=(3, 2, 6)[count.selected_index],
            guided=navigation.selected_index == 1,
            validation=validation.items[validation.selected_index],
            embedded_grid=grid.selected_index == 1,
        )


def main() -> None:
    """Open the interactive design comparison."""
    StepperWindow().run_blocking()


if __name__ == "__main__":
    main()
