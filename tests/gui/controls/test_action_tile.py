"""Tests for the action tile's layout and theme rules, painted on a host that records drawing calls."""

import pytest
from pysual import Painter, Rect, get_theme
from pysual.painting import resolve_style

from sustentable.gui.controls.action_tile import ActionTile, with_action_tile

THEME = with_action_tile(get_theme("modern_dark"))
WIDTH, HEIGHT = 340.0, 260.0
# The card's padding on each side, so the text can use the rest of its width.
TEXT_WIDTH = WIDTH - 2 * 24
CHAR_WIDTH = 0.6


class _RecordingHost:
    """Record drawn text and rectangles, measuring text as a fixed width per character."""

    capabilities = frozenset()
    size = (WIDTH, HEIGHT)

    def __init__(self) -> None:
        """Start with no recorded drawing."""
        self.rects: list[tuple[Rect, str, float]] = []
        self.texts: list[tuple[str, float, str, float]] = []

    def clip(self, rect: Rect | None) -> None:
        """Ignore clipping, which these tests do not check."""

    def rect(self, rect: Rect, fill: str, radius: float = 0, border: str = "", border_width: float = 1) -> None:
        """Record a filled rectangle; the card and the icon badge are drawn this way."""
        self.rects.append((rect, fill, radius))

    def text(self, text: str, x: float, y: float, color: str, size: float, mono: bool = False) -> None:
        """Record the drawn text with its left edge, color and size."""
        self.texts.append((text, x, color, size))

    def measure(self, text: str, size: float, mono: bool = False) -> tuple[float, float]:
        """Give every character the same width."""
        return len(text) * size * CHAR_WIDTH, size * 1.3

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1) -> None:
        """Ignore the icon's strokes, which these tests do not check."""


def _paint(tile: ActionTile) -> _RecordingHost:
    """Paint ``tile`` on a 340 x 260 recording host."""
    host = _RecordingHost()
    tile.paint(Painter(host, Rect(0, 0, *host.size), tile.effective_theme, tile))
    return host


def _tile(description: str = "Start from a CSV, Excel or SQLite file to create a new model.") -> ActionTile:
    """A detached tile styled with the application's theme rules."""
    return ActionTile(text="Load dataset", description=description, icon="grid", theme_override=THEME)


def test_title_and_wrapped_description_are_centered_in_the_card() -> None:
    """The title comes first, and the description wraps at spaces into lines that each fit and are centered."""
    tile = _tile()
    texts = _paint(tile).texts
    (title, *_), *lines = texts

    assert title == "Load dataset"
    assert len(lines) == 2 and " ".join(text for text, *_ in lines) == tile.description
    for text, x, _, size in texts:
        width = len(text) * size * CHAR_WIDTH
        assert width <= TEXT_WIDTH
        assert x + width / 2 == pytest.approx(WIDTH / 2)


def test_a_long_description_is_cut_with_an_ellipsis_after_three_lines() -> None:
    """Text that does not fit in three lines ends the last one with an ellipsis instead of overflowing the card."""
    lines = [text for text, *_ in _paint(_tile(" ".join(["description"] * 30))).texts[1:]]

    assert len(lines) == 3 and lines[-1].endswith("…")


def test_badge_and_text_colors_follow_the_theme_and_the_enabled_state() -> None:
    """The badge uses the accent color and the description is muted; a disabled tile mutes all of it."""
    t = THEME.tokens
    tile = _tile()

    def badges(host: _RecordingHost, fill: str) -> list[Rect]:
        return [rect for rect, color, radius in host.rects if color == fill and radius == rect.width / 2]

    host = _paint(tile)
    [badge] = badges(host, t.accent)
    assert badge.x + badge.width / 2 == pytest.approx(WIDTH / 2)
    assert [color for _, _, color, _ in host.texts] == [t.foreground, t.muted, t.muted]

    tile.enabled = False
    host = _paint(tile)
    assert not badges(host, t.accent) and len(badges(host, t.raised)) == 1
    assert {color for _, _, color, _ in host.texts} == {t.muted}


def test_hover_and_pressed_raise_the_card_with_an_accent_border() -> None:
    """The card reacts to the pointer like a button, without changing the size of its text."""
    t = THEME.tokens
    tile = _tile()
    normal, hover, pressed = (resolve_style(tile, state=state) for state in ("normal", "hover", "pressed"))

    assert (normal.fill, hover.fill, pressed.fill) == (t.surface, t.raised, t.selection)
    assert normal.border == t.border and hover.border == pressed.border == t.accent
    assert normal.font_size == hover.font_size == pressed.font_size
