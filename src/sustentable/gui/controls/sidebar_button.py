"""A borderless Pysual button that only shows a background while hovered or pressed."""

from dataclasses import replace

from pysual import Button, Rule, Style, Theme

TRANSPARENT = "#00000000"


class SidebarButton(Button):
    """Button with no background or border; it is only highlighted on hover.

    Its look comes from the rules that ``with_sidebar_button`` adds to the window theme.
    """

    _style_kind = "SidebarButton"


def with_sidebar_button(theme: Theme) -> Theme:
    """Return a copy of ``theme`` with the normal, hover, and pressed rules for ``SubtleButton``."""
    t = theme.tokens
    rules = (
        Rule("SidebarButton", state="normal", style=Style(fill=TRANSPARENT, border=TRANSPARENT, foreground=t.muted, radius=6)),
        Rule("SidebarButton", state="hover", style=Style(fill=t.raised, border=TRANSPARENT, foreground=t.foreground)),
        Rule("SidebarButton", state="pressed", style=Style(fill=t.border, border=TRANSPARENT, foreground=t.foreground)),
    )
    return replace(theme, rules=theme.rules + rules)
