"""One configurable Pysual workflow, shared by all prototype alternatives."""

from functools import partial

from pysual import (
    Button,
    ChangeEvent,
    ClickEvent,
    Container,
    Label,
    ScrollArea,
    Step,
    Stepper,
    UiEvent,
    ValidationEvent,
)

from sustentable.gui.controls.sidebar_button import SidebarButton

from .columns_step import ColumnsForm
from .dataset_step import DatasetForm
from .preview import SAMPLE_ROWS, sample_grid
from .split_step import SplitForm


class Workflow(Container):
    """Compare orientation, grouping, navigation, and validation using the same controls."""

    def __init__(
        self, *, orientation: str, steps: int, guided: bool, validation: str, embedded_grid: bool = False, **properties: object
    ) -> None:
        """Build only the selected alternative, using native stepper navigation."""
        super().__init__(layout="stack", spacing=8, **properties)
        self.validation = validation
        self.rail = orientation == "rail"
        horizontal = orientation == "horizontal"
        self.body = Container(parent=self, layout="stack", direction="vertical" if horizontal else "horizontal", spacing=16, flex=1)
        width = 184 if self.rail else 360 if not horizontal and not embedded_grid else None
        fill = embedded_grid and not self.rail
        setup = (Container if fill else ScrollArea)(parent=self.body, layout="stack", width=width, spacing=8, flex=1 if fill else 0)
        self.wizard = Stepper(
            parent=setup, orientation="horizontal" if horizontal else "vertical", linear=guided, header_height=48, flex=1 if fill else 0
        )
        workspace = Container(parent=self.body, layout="stack", spacing=8, flex=1) if self.rail else setup
        footer = Container(parent=self, layout="dock", height=36, spacing=8)
        self.advance = Button(parent=footer, text="Continue", dock="right", width=132)
        self.back = SidebarButton(parent=footer, text="Back", dock="right", width=80)
        definitions = (
            [("Data & columns", (DatasetForm, ColumnsForm)), ("Split", (SplitForm,))]
            if steps == 2
            else [("Dataset", (DatasetForm,)), ("Columns", (ColumnsForm,)), ("Split", (SplitForm,))]
        )
        if steps == 6:
            definitions += [(title, ()) for title in ("Preprocess", "Model", "Predict")]
        self.forms = {}
        self.pages = {}
        self.grids = []
        for title, types in definitions:
            step = Step(
                parent=self.wizard, title=title, optional=not types, height=0 if self.rail or embedded_grid or not types else None, padding=0
            )
            page = Container(parent=workspace, layout="stack", visible=False, flex=1 if embedded_grid else 0) if self.rail else step
            across = not embedded_grid and (horizontal or self.rail)
            page.update(padding=8 if types else 0, spacing=8, direction="horizontal" if across or embedded_grid else "vertical")
            field_host = ScrollArea(parent=page, layout="stack", spacing=8, width=280) if embedded_grid and types else page
            fields = tuple(kind(parent=field_host, width=300 if across else None) for kind in types)
            self.forms[step] = fields
            self.pages[step] = page
            if embedded_grid and types:
                self.grids.append(sample_grid(page, flex=1))
            if validation != "None" and fields:
                step.validating.connect(self._validate)
            for field in fields:
                for control in field.editors:
                    channel = control.click if isinstance(control, Button) else control.changed
                    channel.connect(partial(self._edited, step))
        if not embedded_grid:
            preview = Container(parent=workspace if self.rail else self.body, layout="stack", spacing=8, flex=1)
            Label(parent=preview, text="Sample preview", height=24)
            self.grids.append(sample_grid(preview, flex=1))
        self.feedback = Label(parent=self, height=24)
        self._refresh()

    def _problem(self, step: Step) -> str:
        return next((field.problem for field in self.forms[step] if field.problem), "")

    def _validate(self, event: ValidationEvent) -> None:
        if problem := self._problem(event.source):
            event.reject(problem)
            self.feedback.text = problem

    def _edited(self, step: Step, event: UiEvent) -> None:
        for later in self.wizard.steps[self.wizard.steps.index(step) :]:
            later.completed = False
        if step.error:
            step.set_validation(self._problem(step))
        if any(isinstance(field, DatasetForm) and field.loaded for fields in self.forms.values() for field in fields):
            for grid in self.grids:
                grid.rows = SAMPLE_ROWS
        self._refresh()

    def wizard_on_changed(self, event: ChangeEvent[int]) -> None:
        """Update the active panel and its actions."""
        self._refresh()

    def back_on_click(self, event: ClickEvent) -> None:
        """Keep the form values when returning to a previous step."""
        self.wizard.previous()

    async def advance_on_click(self, event: ClickEvent) -> None:
        """Validate only when requested, then use the built-in navigation."""
        if not self.advance.enabled:
            return
        step = self.wizard.selected_step
        if self.forms[step] and self.validation != "None" and not await step.validate():
            self._refresh()
            return
        if self.forms[step]:
            step.completed = True
        if not self.wizard.next():
            self.feedback.text = "Demo complete · no data processing performed."
        else:
            self._refresh()

    def _refresh(self) -> None:
        step = self.wizard.selected_step
        if self.rail:
            for key, page in self.pages.items():
                page.visible = key is step
        problem = self._problem(step) if self.validation != "None" else ""
        self.feedback.text = (
            problem or "Ready to continue."
            if self.validation == "Live hints"
            else step.error or ("Empty placeholder." if not self.forms[step] else "Click a step header or use Back / Continue.")
        )
        self.advance.enabled = self.validation != "Live hints" or not problem
        self.advance.text = "Finish demo" if self.wizard.selected_index == len(self.wizard.steps) - 1 else "Continue"
        self.back.enabled = self.wizard.selected_index > 0
