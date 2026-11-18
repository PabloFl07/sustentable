"""A large Pysual button drawn as a card with an icon badge, a title and a short description."""

from dataclasses import replace
from typing import ClassVar, Literal

from pysual import Button, Dirty, Painter, Rect, Rule, Style, Theme, prop
from pysual.host import Host

from sustentable.gui.controls.sidebar_button import TRANSPARENT

_STYLE_KIND = "ActionTile"
_BADGE_SIZE = 76
_ICON_SIZE = 38
_GAP = 16
_MAX_DESCRIPTION_LINES = 3


class ActionTile(Button):
    """Button drawn as a card: a round icon badge above a large title and a muted, wrapped description.

    It emits ``click`` like any other button. The description wraps at spaces into at most three centered lines;
    longer text is cut with an ellipsis. Its look comes from the rules that ``with_action_tile`` adds to the window theme.
    """

    description: str = prop(default="", affects=Dirty.MEASURE | Dirty.PAINT)
    style_parts: ClassVar[tuple[str, ...]] = ("body", "icon", "description")
    _style_kind = _STYLE_KIND

    def measure(self, host: Host) -> tuple[float, float]:
        """Prefer a large card; layouts usually give it a fixed size."""
        return 320, 240

    def paint(self, painter: Painter, /) -> None:
        """Draw the card, then the badge, title and description as one block centered in it."""
        body = painter.body()
        badge, detail = painter.style("icon"), painter.style("description")
        area = painter.content_rect()
        title_height = painter.measure("Ag", size=body.font_size, font_family=body.font_family)[1]
        line_height = painter.measure("Ag", size=detail.font_size, font_family=detail.font_family)[1] + 4
        lines = _wrap(painter, self.description, area.width, detail.font_size, detail.font_family)
        if len(lines) > _MAX_DESCRIPTION_LINES:
            # The caption elides the joined remainder, so the last line ends with "…".
            lines = [*lines[: _MAX_DESCRIPTION_LINES - 1], " ".join(lines[_MAX_DESCRIPTION_LINES - 1 :])]

        block = _BADGE_SIZE + _GAP + title_height + (_GAP / 2 + line_height * len(lines) if lines else 0)
        top = area.y + max(0, (area.height - block) / 2)
        left = area.x + (area.width - _BADGE_SIZE) / 2
        painter.rect(Rect(left, top, _BADGE_SIZE, _BADGE_SIZE), badge.fill, radius=_BADGE_SIZE / 2)
        if self.icon:
            inset = (_BADGE_SIZE - _ICON_SIZE) / 2
            painter.icon(self.icon, left + inset, top + inset, size=_ICON_SIZE, color=badge.foreground)

        top += _BADGE_SIZE + _GAP
        title = Rect(area.x, top, area.width, title_height)
        painter.caption(self.text, title, align="center", color=body.foreground, size=body.font_size, font_family=body.font_family)
        top += title_height + _GAP / 2
        for line in lines:
            row = Rect(area.x, top, area.width, line_height)
            painter.caption(line, row, align="center", color=detail.foreground, size=detail.font_size, font_family=detail.font_family)
            top += line_height


def with_action_tile(theme: Theme) -> Theme:
    """Return a copy of ``theme`` with the card, icon badge and description rules for ``ActionTile``."""
    t = theme.tokens
    # Glossy themes give buttons a top highlight line, which a large card shows as a second border.
    card = Style(fill=t.surface, border=t.border, foreground=t.foreground, radius=16, font_size=22, padding=24, highlight=TRANSPARENT)
    rules = (
        Rule(_STYLE_KIND, state="normal", style=card),
        Rule(_STYLE_KIND, state="hover", style=Style(fill=t.raised, border=t.accent)),
        Rule(_STYLE_KIND, state="pressed", style=Style(fill=t.selection, border=t.accent)),
        Rule(_STYLE_KIND, state="disabled", style=Style(fill=t.surface, border=t.border, foreground=t.muted)),
        Rule(_STYLE_KIND, part="icon", style=Style(fill=t.accent, foreground=t.accent_text)),
        Rule(_STYLE_KIND, part="icon", state="disabled", style=Style(fill=t.raised, foreground=t.muted)),
        Rule(_STYLE_KIND, part="description", style=Style(foreground=t.muted, font_size=14)),
    )
    return replace(theme, rules=theme.rules + rules)


def _wrap(painter: Painter, text: str, width: float, size: float, font_family: Literal["ui", "mono"]) -> list[str]:
    """Break ``text`` at spaces into lines no wider than ``width``; a word wider than that gets a line of its own."""
    lines: list[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and painter.measure(candidate, size=size, font_family=font_family)[0] <= width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines
