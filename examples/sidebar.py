from dataclasses import replace

from pysual import App, Button, CheckBox, Container, Label, RadioButton, Rule, ScrollArea, Style


class BotonSutil(Button):
    """Button with no background or border; it is only highlighted on hover."""

    _style_kind = "BotonSutil"


def con_boton_sutil(theme):
    t = theme.tokens
    transparente = "#00000000"
    reglas = (
        Rule("BotonSutil", state="normal", style=Style(fill=transparente, border=transparente, foreground=t.muted, radius=6)),
        Rule("BotonSutil", state="hover", style=Style(fill=t.raised, border=transparente, foreground=t.foreground)),
        Rule("BotonSutil", state="pressed", style=Style(fill=t.border, border=transparente, foreground=t.foreground)),
    )
    return replace(theme, rules=theme.rules + reglas)


class Regresion(App):
    def build(self):
        self.title = "Regression"
        self.width, self.height = 820, 540
        self.layout, self.padding, self.spacing = "stack", 12, 12
        self.theme = con_boton_sutil(self.theme)
        fondo = self.theme.tokens.surface

        self.cuerpo = Container(layout="stack", direction="horizontal", flex=1, spacing=12)

        # ── Sidebar ──────────────────────────────────────────────
        self.sidebar = ScrollArea(parent=self.cuerpo, layout="stack", width=380, padding=12, spacing=6, background=fondo)

        self.cabecera = Container(parent=self.sidebar, layout="stack", direction="horizontal", height=28)
        Label(parent=self.cabecera, text="COLUMNS", font_size=24, flex=1, background=fondo)
        self.ocultar = BotonSutil(parent=self.cabecera, icon="close", width=28, height=28, tooltip="Hide panel")

        Label(parent=self.sidebar, text="Input (X)", height=22, font_size=20, background=fondo)
        self.grupo_x = Container(parent=self.sidebar, layout="stack", spacing=4)
        Label(parent=self.sidebar, text="Output (Y)", height=22, font_size=20, background=fondo)
        self.grupo_y = Container(parent=self.sidebar, layout="stack", spacing=4)

        # Mode header (outside the box)
        self.modo = Label(parent=self.sidebar, text="", height=24, font_size=20, background=fondo)
        # Box with the formula
        cuadro = self.theme.tokens.raised
        self.cuadro_modo = Container(parent=self.sidebar, layout="stack", padding=20, background=cuadro)
        self.formula = Label(parent=self.cuadro_modo, text="", height=50, font_size=18, word_wrap=True, background=cuadro)

        self.confirmar = Button(parent=self.sidebar, text="Confirm selection", icon="check", height=36, enabled=False)

        # ── Content ──────────────────────────────────────────────
        self.contenido = Container(parent=self.cuerpo, layout="stack", flex=1, padding=12, spacing=8)
        self.mostrar = BotonSutil(parent=self.contenido, icon="menu", width=28, height=28, tooltip="Show panel", visible=False)
        Label(parent=self.contenido, text="Chart / results go here", font_size=18, height=30)

        self._checks_x: dict[str, CheckBox] = {}
        self._radios_y: dict[str, RadioButton] = {}
        self.cargar_columnas(["area", "rooms", "age", "price"])

    # ── Show / hide ──────────────────────────────────────────────
    def ocultar_on_click(self, event):
        self.sidebar.visible = False
        self.mostrar.visible = True

    def mostrar_on_click(self, event):
        self.sidebar.visible = True
        self.mostrar.visible = False

    # ── Columns ──────────────────────────────────────────────────
    def cargar_columnas(self, columnas):
        for control in [*self._checks_x.values(), *self._radios_y.values()]:
            control.destroy()
        self._checks_x.clear()
        self._radios_y.clear()

        fondo = self.theme.tokens.surface
        for col in columnas:
            check = CheckBox(parent=self.grupo_x, text=col, height=30, background=fondo, font_size=18)
            check.changed.connect(self.on_seleccion)
            self._checks_x[col] = check

            radio = RadioButton(parent=self.grupo_y, text=col, group="output", height=30, background=fondo, font_size=18)
            radio.changed.connect(self.on_seleccion)
            self._radios_y[col] = radio

        self.actualizar_modo()

    @property
    def columnas_x(self) -> list[str]:
        return [c for c, check in self._checks_x.items() if check.checked]

    @property
    def columna_y(self) -> str | None:
        return next((c for c, radio in self._radios_y.items() if radio.checked), None)

    def obtener_seleccion(self) -> tuple[list[str], str | None]:
        """Return (input columns, output column)."""
        return self.columnas_x, self.columna_y

    def confirmar_on_click(self, event):
        x, y = self.obtener_seleccion()
        self.seleccion_confirmada = (x, y)
        print("X:", x, "| Y:", y)  # call your regression function here

    def on_seleccion(self, event):
        self.actualizar_modo()

    def actualizar_modo(self):
        y = self.columna_y
        for col, check in self._checks_x.items():
            check.enabled = col != y
            if col == y and check.checked:
                check.checked = False

        x = self.columnas_x
        if not x or y is None:
            self.modo.text = "Choose at least one X and one Y"
            self.formula.text = ""
            self.cuadro_modo.visible = False
            self.confirmar.enabled = False
            return

        self.modo.text = "Simple regression" if len(x) == 1 else "Multiple regression"
        self.formula.text = f"{y} ~ {' + '.join(x)}"
        self.cuadro_modo.visible = True
        self.confirmar.enabled = True


if __name__ == "__main__":
    Regresion().run_blocking()
