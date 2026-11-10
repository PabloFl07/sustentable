"""A Pysual scatter chart: point series and straight lines over labeled numeric axes, with a legend."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, isfinite, log10
from typing import ClassVar, Literal

from pysual import Control, Dirty, Painter, Rect, prop
from pysual.host import Host

# Painting every point of a large dataset slows each frame without changing the picture much.
MAX_POINTS = 2_000
_MARKER_SIZE = 7
_MARGIN = 12
_ROW = 20
_TICK_SIZE = 11
_TITLE_SIZE = 13
_COLOR = re.compile(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?")

type Point = tuple[float, float]


@dataclass(frozen=True)
class PointSeries:
    """Points drawn as markers of one shape and color, and named ``title`` in the legend."""

    title: str
    points: tuple[Point, ...]
    color: str
    shape: Literal["circle", "square"] = "circle"


@dataclass(frozen=True)
class LineSeries:
    """A straight line from ``start`` to ``end``, named ``title`` in the legend."""

    title: str
    start: Point
    end: Point
    color: str


class ScatterChart(Control):
    """Draw point series and straight lines over numeric axes, with grid lines, axis titles and a legend.

    ``set_data`` replaces what is drawn and ``clear`` empties the chart, which then shows ``empty_text``.
    Series longer than ``MAX_POINTS`` are drawn with that many evenly spaced points, but the axes still cover every
    point. The y axis title is written above the axis, because Pysual cannot rotate text.
    Colors are ``#RRGGBB`` or ``#RRGGBBAA``, so markers can be translucent where they overlap.
    """

    cache_paint: bool = prop(default=True)
    x_label: str = prop(default="", affects=Dirty.PAINT)
    y_label: str = prop(default="", affects=Dirty.PAINT)
    empty_text: str = prop(default="No data", affects=Dirty.PAINT)
    style_fallbacks: ClassVar[tuple[str, ...]] = ("LineChart",)
    style_parts: ClassVar[tuple[str, ...]] = ("body", "axis", "legend")

    def __init__(self, **properties: object) -> None:
        """Create an empty chart; ordinary Control properties such as ``flex`` are accepted."""
        super().__init__(**properties)
        self._points: tuple[PointSeries, ...] = ()
        self._lines: tuple[LineSeries, ...] = ()
        self._drawn: tuple[tuple[Point, ...], ...] = ()
        self._extent = (0.0, 1.0, 0.0, 1.0)

    @property
    def point_series(self) -> tuple[PointSeries, ...]:
        """Point series as they were passed to ``set_data``."""
        return self._points

    @property
    def line_series(self) -> tuple[LineSeries, ...]:
        """Lines as they were passed to ``set_data``."""
        return self._lines

    @property
    def drawn_points(self) -> tuple[tuple[Point, ...], ...]:
        """Points painted for each point series, after thinning the long ones."""
        return self._drawn

    def set_data(self, points: Sequence[PointSeries], lines: Sequence[LineSeries] = ()) -> None:
        """Replace every series and line, keeping the previous ones if the new data is invalid.

        Raises:
            ValueError: If a coordinate is not a finite number or a color is not ``#RRGGBB`` or ``#RRGGBBAA``.
        """
        points, lines = tuple(points), tuple(lines)
        coordinates = [point for series in points for point in series.points]
        coordinates += [end for line in lines for end in (line.start, line.end)]
        if not all(isfinite(x) and isfinite(y) for x, y in coordinates):
            raise ValueError("Chart coordinates must be finite numbers.")
        for item in (*points, *lines):
            if not _COLOR.fullmatch(item.color):
                raise ValueError(f"Expected a #RRGGBB or #RRGGBBAA color, got {item.color!r}.")
        self._points, self._lines = points, lines
        self._drawn = tuple(_thin(series.points) for series in points)
        self._extent = _extent(coordinates)
        self.invalidate()

    def clear(self) -> None:
        """Remove every series and line."""
        self.set_data(())

    def measure(self, host: Host) -> tuple[float, float]:
        """Prefer a medium chart; layouts usually stretch it with ``flex``."""
        return 480, 320

    def paint(self, painter: Painter, /) -> None:
        """Draw the grid, the axes and their titles, the series and the legend."""
        body = painter.body()
        axis, legend = painter.style("axis"), painter.style("legend")
        family = axis.font_family

        def text_width(text: str, size: float = _TICK_SIZE) -> float:
            return painter.measure(text, size=size, font_family=family)[0]

        def write(text: str, x: float, y: float, size: float, color: str) -> None:
            painter.text(text, x, y, color=color, size=size, font_family=family)

        if not self._points and not self._lines:
            write(self.empty_text, _MARGIN, _MARGIN, _TITLE_SIZE, axis.foreground)
            return

        # From top to bottom: y axis title, plot, x tick labels, x axis title and legend.
        top = _MARGIN + (_ROW + 4 if self.y_label else 0) + 6
        bottom = painter.height - _MARGIN - _ROW * (2 + bool(self.x_label))
        if bottom - top < _ROW:
            return
        x_low, x_high, y_low, y_high = self._extent
        y_ticks, y_texts = _ticks(y_low, y_high, max(2, int((bottom - top) // 48)))
        left = _MARGIN + max(map(text_width, y_texts)) + 8
        plot = Rect(left, top, painter.width - _MARGIN - 8 - left, bottom - top)
        if plot.width < _ROW:
            return
        x_count = max(2, int(plot.width // 96))
        x_ticks, x_texts = _ticks(x_low, x_high, x_count)
        # Fewer ticks while the widest label would overlap the next one.
        while x_count > 2 and max(map(text_width, x_texts)) + 16 > plot.width / (len(x_ticks) - 1):
            x_count -= 1
            x_ticks, x_texts = _ticks(x_low, x_high, x_count)

        x_min, x_span = x_ticks[0], x_ticks[-1] - x_ticks[0]
        y_min, y_span = y_ticks[0], y_ticks[-1] - y_ticks[0]

        def to_x(value: float) -> float:
            return plot.x + plot.width * (value - x_min) / x_span

        def to_y(value: float) -> float:
            return plot.bottom - plot.height * (value - y_min) / y_span

        for value, text in zip(y_ticks, y_texts, strict=True):
            y = to_y(value)
            painter.line(plot.x, y, plot.right, y, axis.border)
            write(text, plot.x - 8 - text_width(text), y - _TICK_SIZE * 0.65, _TICK_SIZE, axis.foreground)
        for value, text in zip(x_ticks, x_texts, strict=True):
            x = to_x(value)
            painter.line(x, plot.y, x, plot.bottom, axis.border)
            width = text_width(text)
            write(text, min(max(0, x - width / 2), painter.width - width), plot.bottom + 5, _TICK_SIZE, axis.foreground)
        painter.line(plot.x, plot.y, plot.x, plot.bottom, axis.foreground)
        painter.line(plot.x, plot.bottom, plot.right, plot.bottom, axis.foreground)

        if self.y_label:
            title = painter.elide(self.y_label, painter.width - 2 * _MARGIN, size=_TITLE_SIZE, font_family=family)
            write(title, _MARGIN, _MARGIN, _TITLE_SIZE, body.foreground)
        if self.x_label:
            title = painter.elide(self.x_label, plot.width, size=_TITLE_SIZE, font_family=family)
            write(title, plot.x + (plot.width - text_width(title, _TITLE_SIZE)) / 2, plot.bottom + _ROW + 4, _TITLE_SIZE, body.foreground)

        for series, drawn in zip(self._points, self._drawn, strict=True):
            for x, y in drawn:
                _marker(painter, to_x(x), to_y(y), series.shape, series.color, _MARKER_SIZE)
        for line in self._lines:
            painter.line(to_x(line.start[0]), to_y(line.start[1]), to_x(line.end[0]), to_y(line.end[1]), line.color, 2)

        # The legend lists the point series and then the lines, on one row below the x axis title.
        x, y = plot.x, painter.height - _MARGIN - _ROW + 3
        for item in (*self._points, *self._lines):
            if isinstance(item, PointSeries):
                _marker(painter, x + 8, y + 8, item.shape, item.color, _MARKER_SIZE + 2)
            else:
                painter.line(x, y + 8, x + 16, y + 8, item.color, 2)
            title = painter.elide(item.title, painter.width - _MARGIN - x - 22, size=_TITLE_SIZE - 1, font_family=family)
            write(title, x + 22, y, _TITLE_SIZE - 1, legend.foreground)
            x += 22 + text_width(title, _TITLE_SIZE - 1) + 24


def _marker(painter: Painter, x: float, y: float, shape: Literal["circle", "square"], color: str, size: float) -> None:
    """Draw a marker of ``size`` pixels centred on (x, y)."""
    box = Rect(x - size / 2, y - size / 2, size, size)
    painter.rect(box, color, radius=size / 2 if shape == "circle" else 0)


def _thin(points: tuple[Point, ...], limit: int = MAX_POINTS) -> tuple[Point, ...]:
    """Keep at most ``limit`` evenly spaced points, starting with the first one."""
    if len(points) <= limit:
        return points
    step = len(points) / limit
    return tuple(points[int(index * step)] for index in range(limit))


def _extent(coordinates: list[Point]) -> tuple[float, float, float, float]:
    """Return the x and y ranges of the points, widened so that no marker sits on the border of the plot."""
    if not coordinates:
        return 0.0, 1.0, 0.0, 1.0
    xs, ys = [x for x, _ in coordinates], [y for _, y in coordinates]
    return *_widen(min(xs), max(xs)), *_widen(min(ys), max(ys))


def _widen(low: float, high: float) -> tuple[float, float]:
    """Add a small margin to a range, or give a single value a range around it."""
    span = high - low
    margin = span * 0.04 if span > 0 else max(abs(low) * 0.1, 1.0)
    return low - margin, high + margin


def _ticks(low: float, high: float, count: int) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Return round tick values from ``low`` or below to ``high`` or above, about ``count`` steps apart, and their labels.

    Steps are 1, 2 or 5 times a power of ten, and every label has the decimals the step needs.
    """
    raw = (high - low) / count
    if isfinite(raw) and raw > 0:
        exponent = floor(log10(raw))
        factor = next(factor for factor in (1, 2, 5, 10) if factor * 10.0**exponent >= raw)
        if factor == 10:
            factor, exponent = 1, exponent + 1
        step = factor * 10.0**exponent
        first, last = floor(low / step), ceil(high / step)
        values = tuple((first + index) * step for index in range(last - first + 1))
        if len(values) >= 2 and values[-1] > values[0]:
            decimals = max(0, -exponent)
            return values, tuple(f"{value:,.{decimals}f}" for value in values)
    # Extreme magnitudes leave no room for round steps, so the axis only shows its ends.
    return (low, high), (f"{low:,g}", f"{high:,g}")
