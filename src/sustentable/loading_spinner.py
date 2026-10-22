"""A small rotating activity ring for work with no measurable completion percentage."""

import asyncio
from math import cos, sin, tau
from typing import ClassVar

from pysual import Control, Dirty, Painter, Rect, prop
from pysual.host import Host


class LoadingSpinner(Control):
    """Draw a themed activity ring; the caller owns and cancels the ``animate`` coroutine."""

    phase: int = prop(default=0, minimum=0, affects=Dirty.PAINT)
    style_fallbacks: ClassVar[tuple[str, ...]] = ("CircularProgress", "ProgressBar")
    style_parts: ClassVar[tuple[str, ...]] = ("body", "track", "fill")

    def measure(self, host: Host) -> tuple[float, float]:
        """Reserve a compact square beside the loading message."""
        return 24, 24

    async def animate(self) -> None:
        """Advance the ring on the UI loop until the owning load cancels this coroutine."""
        self.phase = 0
        while True:
            await asyncio.sleep(1 / 12)
            self.phase = (self.phase + 1) % 12

    def paint(self, painter: Painter, /) -> None:
        """Draw the track and rotating arc using the theme's progress colors."""
        super().paint(painter)
        diameter = min(painter.width, painter.height) - 4
        if diameter <= 0:
            return
        thickness = min(2.5, diameter / 2)
        radius = (diameter - thickness) / 2
        cx, cy = painter.width / 2, painter.height / 2
        track, fill = painter.style("track"), painter.style("fill")
        color = fill.fill or fill.foreground
        circle = tuple((cx + radius * cos(tau * step / 36), cy + radius * sin(tau * step / 36)) for step in range(37))
        painter.lines(circle, track.fill or track.foreground, thickness)
        start = tau * self.phase / 12
        arc = tuple((cx + radius * cos(start + tau * step / 32), cy + radius * sin(start + tau * step / 32)) for step in range(25))
        painter.lines(arc, color, thickness)
        for x, y in (arc[0], arc[-1]):
            painter.rect(Rect(x - thickness / 2, y - thickness / 2, thickness, thickness), color, radius=thickness / 2)
