"""A Pysual scatter chart for a regression fit.

A flat chart draws point series and straight lines on two labeled axes. Two inputs are drawn with the output as the
height of a regression plane: drag to rotate, Shift-drag to pan and scroll to zoom. Axis titles stay horizontal,
because Pysual cannot rotate text. Series longer than ``MAX_POINTS`` are drawn with that many evenly spaced points,
but the axes still cover every point.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil, cos, floor, hypot, inf, isfinite, log10, radians, sin
from typing import ClassVar, Literal

from pysual import Control, Dirty, Painter, Rect, prop
from pysual.host import Host, Input

# Painting every point of a large dataset slows each frame without changing the picture much.
MAX_POINTS = 2_000
# Cells along each input of the regression plane.
PLANE_STEPS = 8
# Camera of a chart that has not been dragged. Azimuth turns around the vertical axis; elevation rises above the inputs.
DEFAULT_AZIMUTH = -60.0
DEFAULT_ELEVATION = 30.0
_MIN_ELEVATION = 12.0
_MAX_ELEVATION = 80.0
# Degrees the view turns for each pixel of a drag.
_DRAG_DEGREES = 0.4
# One wheel step toward the user multiplies the size by this. The size stays between the two limits.
_ZOOM_STEP = 1.2
_MIN_ZOOM = 0.5
_MAX_ZOOM = 5.0
_MARKER_SIZE = 7
# Small markers keep a dense cloud from covering the plane. The legend draws them a little larger.
_VOLUME_MARKER = 4
_MARGIN = 12
_ROW = 20
_TICK_SIZE = 11
_TITLE_SIZE = 13
_TICK_COUNT = 4
# Share of the plot the cube fills, leaving room for the titles around it.
_FIT = 0.72
_VIEW_HINT = "Drag to rotate · Shift-drag to pan · Scroll to zoom"
_COLOR = re.compile(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?")

type Point = tuple[float, float]
type Point3 = tuple[float, float, float]
type _Projected = tuple[float, float, float]
type _Drag = tuple[int, float, float, float, float, float, float, Literal["rotate", "pan"]]


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


@dataclass(frozen=True)
class PointSeries3D:
    """Points of a three-dimensional series, drawn as markers and named ``title`` in the legend."""

    title: str
    points: tuple[Point3, ...]
    color: str
    shape: Literal["circle", "square"] = "circle"


@dataclass(frozen=True)
class PlaneSeries:
    """A rectangular regression plane drawn as a grid, and named ``title`` in the legend.

    ``corners`` are ``(x low, y low)``, ``(x high, y low)``, ``(x high, y high)`` and ``(x low, y high)``.
    """

    title: str
    corners: tuple[Point3, Point3, Point3, Point3]
    color: str


class ScatterChart(Control):
    """Draw a flat fit, or a two-input fit as a plane, over labeled numeric axes.

    ``set_data`` shows points and straight lines. ``set_volume`` shows points and a regression plane and returns to
    the default view: drag orbits it, Shift-drag pans it and the wheel zooms it. ``clear`` empties the chart, which
    then shows ``empty_text``. The elevation stays between a shallow angle and a steep one, so the output axis keeps
    pointing up. Colors are ``#RRGGBB`` or ``#RRGGBBAA``, so flat markers can be translucent where they overlap.
    """

    cache_paint: bool = prop(default=True)
    x_label: str = prop(default="", affects=Dirty.PAINT)
    y_label: str = prop(default="", affects=Dirty.PAINT)
    z_label: str = prop(default="", affects=Dirty.PAINT)
    azimuth: float = prop(default=DEFAULT_AZIMUTH, affects=Dirty.PAINT)
    elevation: float = prop(default=DEFAULT_ELEVATION, affects=Dirty.PAINT)
    zoom: float = prop(default=1.0, affects=Dirty.PAINT)
    pan_x: float = prop(default=0.0, affects=Dirty.PAINT)
    pan_y: float = prop(default=0.0, affects=Dirty.PAINT)
    empty_text: str = prop(default="No data", affects=Dirty.PAINT)
    style_fallbacks: ClassVar[tuple[str, ...]] = ("LineChart",)
    style_parts: ClassVar[tuple[str, ...]] = ("body", "axis", "legend")

    def __init__(self, **properties: object) -> None:
        """Create an empty chart; ordinary Control properties such as ``flex`` are accepted."""
        super().__init__(**properties)
        self._points: tuple[PointSeries, ...] = ()
        self._lines: tuple[LineSeries, ...] = ()
        self._volume: tuple[PointSeries3D, ...] = ()
        self._plane: PlaneSeries | None = None
        self._spatial = False
        self._drawn: tuple[tuple[Point, ...], ...] | tuple[tuple[Point3, ...], ...] = ()
        self._extent = (0.0, 1.0, 0.0, 1.0)
        self._span = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        self._drag: _Drag | None = None

    @property
    def point_series(self) -> tuple[PointSeries, ...] | tuple[PointSeries3D, ...]:
        """Point series of the fit currently shown."""
        return self._volume if self._spatial else self._points

    @property
    def line_series(self) -> tuple[LineSeries, ...]:
        """Lines of the flat fit. Empty while a plane is shown."""
        return self._lines

    @property
    def plane(self) -> PlaneSeries | None:
        """The regression plane of the fit currently shown, or ``None`` for a flat fit."""
        return self._plane

    @property
    def drawn_points(self) -> tuple[tuple[Point, ...], ...] | tuple[tuple[Point3, ...], ...]:
        """Points painted for each point series, after thinning the long ones."""
        return self._drawn

    def set_data(self, points: Sequence[PointSeries], lines: Sequence[LineSeries] = ()) -> None:
        """Show a flat fit, replacing whatever was drawn, and keep the previous one if the new data is invalid.

        Raises:
            ValueError: If a coordinate is not a finite number or a color is not ``#RRGGBB`` or ``#RRGGBBAA``.
        """
        points, lines = tuple(points), tuple(lines)
        coordinates = [point for series in points for point in series.points]
        coordinates += [end for line in lines for end in (line.start, line.end)]
        if not all(isfinite(x) and isfinite(y) for x, y in coordinates):
            raise ValueError("Chart coordinates must be finite numbers.")
        for item in (*points, *lines):
            _check_color(item.color)
        self._show_flat(points, lines, coordinates)

    def set_volume(self, points: Sequence[PointSeries3D], plane: PlaneSeries | None = None) -> None:
        """Show a three-dimensional fit from the default angle, keeping the previous one if the new data is invalid.

        Raises:
            ValueError: If a coordinate is not a finite number, a marker is not a circle or a square, the plane does
                not have four corners, or a color is not ``#RRGGBB`` or ``#RRGGBBAA``.
        """
        points = tuple(points)
        _validate_volume(points, plane)
        self._volume, self._plane = points, plane
        self._points, self._lines = (), ()
        self._spatial = True
        self._drawn = tuple(_thin(series.points) for series in points)
        self._span = _volume_extent(points, plane)
        self._reset_view()
        self.invalidate()

    def clear(self) -> None:
        """Remove every series and return to the default view."""
        self.set_data(())

    def measure(self, host: Host) -> tuple[float, float]:
        """Prefer a medium chart; layouts usually stretch it with ``flex``."""
        return 480, 320

    def paint(self, painter: Painter, /) -> None:
        """Draw the fit currently shown."""
        if self._spatial:
            self._paint_volume(painter)
        else:
            self._paint_flat(painter)

    def handle_input(self, event: Input, /) -> bool | None:
        """Orbit, pan or zoom a three-dimensional fit. A flat chart leaves the wheel to its parent."""
        if not self._spatial:
            return None
        if event.kind == "wheel":
            self.zoom = min(_MAX_ZOOM, max(_MIN_ZOOM, self.zoom * _ZOOM_STEP ** (-event.delta)))
            return True
        if event.kind == "pointer_down" and event.button == 1 and self._drag is None:
            mode: Literal["rotate", "pan"] = "pan" if event.shift else "rotate"
            self._drag = (event.pointer_id, event.x, event.y, self.azimuth, self.elevation, self.pan_x, self.pan_y, mode)
            return True
        if self._drag is None or event.pointer_id != self._drag[0]:
            return None
        if event.kind == "pointer_cancel":
            self._drag = None
            return True
        if event.kind not in ("pointer_move", "pointer_up"):
            return None
        _, x0, y0, azimuth, elevation, pan_x, pan_y, mode = self._drag
        if mode == "pan":
            self.pan_x, self.pan_y = _limit_pan(pan_x + (event.x - x0), pan_y + (event.y - y0), self._rect.width, self._rect.height, self.zoom)
        else:
            self.azimuth = azimuth + (event.x - x0) * _DRAG_DEGREES
            self.elevation = min(_MAX_ELEVATION, max(_MIN_ELEVATION, elevation - (event.y - y0) * _DRAG_DEGREES))
        if event.kind == "pointer_up":
            self._drag = None
        return True

    def _show_flat(self, points: tuple[PointSeries, ...], lines: tuple[LineSeries, ...], coordinates: list[Point]) -> None:
        """Store a validated flat fit and look at the next one from the default angle."""
        self._points, self._lines = points, lines
        self._volume, self._plane = (), None
        self._spatial = False
        self._drawn = tuple(_thin(series.points) for series in points)
        self._extent = _extent(coordinates)
        self._reset_view()
        self.invalidate()

    def _reset_view(self) -> None:
        """Drop a drag and look at the fit from the front, at its original size and position."""
        self._drag = None
        self.azimuth, self.elevation = DEFAULT_AZIMUTH, DEFAULT_ELEVATION
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0

    def _paint_flat(self, painter: Painter) -> None:
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

    def _paint_volume(self, painter: Painter) -> None:
        """Draw the box, the plane, the points, the axis titles and the legend."""
        body = painter.body()
        axis, legend = painter.style("axis"), painter.style("legend")
        family = axis.font_family

        def text_width(text: str, size: float = _TICK_SIZE) -> float:
            return painter.measure(text, size=size, font_family=family)[0]

        def write(text: str, x: float, y: float, size: float, color: str) -> None:
            painter.text(text, x, y, color=color, size=size, font_family=family)

        if not self._volume and self._plane is None:
            write(self.empty_text, _MARGIN, _MARGIN, _TITLE_SIZE, axis.foreground)
            return
        frame = _frame(painter.width, painter.height)
        if frame is None:
            return

        axes = tuple(_ticks(low, high, _TICK_COUNT) for low, high in self._span)
        spans = tuple((values[0], values[-1]) for values, _ in axes)
        project = _projector(spans, self.azimuth, self.elevation, frame, self.zoom, self.pan_x, self.pan_y)
        center = project(_midpoint(spans))
        foreground = axis.foreground
        border = axis.border or foreground
        edges = _box_edges(spans)
        front = _front_edges(edges, project)
        for edge in sorted((edge for edge in edges if edge not in front), key=lambda edge: project(_midpoint_edge(edge))[2]):
            _draw_edge(painter, edge, project, border, 1)
        if self._plane is not None:
            color = self._plane.color
            for line in sorted(_plane_polylines(self._plane.corners, PLANE_STEPS), key=lambda line: sum(project(point)[2] for point in line)):
                painter.lines((project(point)[:2] for point in line), color, 1.5)
        for edge in front:
            _draw_edge(painter, edge, project, foreground, 1.5)
        for series, drawn in sorted(zip(self._volume, self._drawn, strict=True), key=lambda item: _series_depth(item[1], project)):
            for point in sorted(drawn, key=lambda point: project(point)[2]):
                screen = project(point)
                _marker(painter, screen[0], screen[1], series.shape, series.color, _VOLUME_MARKER)

        for axis_index, edge in enumerate(front):
            self._paint_axis(
                painter, write, text_width, axes[axis_index], edge, axis_index, project, center, foreground, body.foreground, family
            )
        self._paint_legend(painter, write, text_width, legend.foreground, family, frame)

    def _paint_axis(
        self,
        painter: Painter,
        write: Callable[[str, float, float, float, str], None],
        text_width: Callable[[str, float], float],
        ticks: tuple[tuple[float, ...], tuple[str, ...]],
        edge: tuple[Point3, Point3],
        axis_index: int,
        project: Callable[[Point3], _Projected],
        center: _Projected,
        tick_color: str,
        title_color: str,
        family: str,
    ) -> None:
        """Draw the tick marks, their labels and the title of one axis, outside the box."""
        values, texts = ticks
        start, end = edge
        if start[axis_index] > end[axis_index]:
            start, end = end, start
        for value, text in zip(values, texts, strict=True):
            screen = project(_along(start, end, axis_index, value))
            mark = _outward(screen[0], screen[1], center, 6)
            painter.line(screen[0], screen[1], mark[0], mark[1], tick_color)
            label = _outward(screen[0], screen[1], center, 10)
            _place_text(write, text, label, text_width(text, _TICK_SIZE), center, painter, _TICK_SIZE, tick_color)
        title = (self.x_label, self.y_label, self.z_label)[axis_index]
        if not title:
            return
        shown = painter.elide(title, painter.width / 3, size=_TITLE_SIZE, font_family=family)
        if not shown:
            return
        middle = project(_midpoint_edge((start, end)))
        label = _outward(middle[0], middle[1], center, 26)
        _place_text(write, shown, label, text_width(shown, _TITLE_SIZE), center, painter, _TITLE_SIZE, title_color)

    def _paint_legend(
        self,
        painter: Painter,
        write: Callable[[str, float, float, float, str], None],
        text_width: Callable[[str, float], float],
        color: str,
        family: str,
        frame: Rect,
    ) -> None:
        """Draw the series, the plane and the hint for rotating, panning and zooming."""
        size = _TITLE_SIZE - 1
        y = painter.height - _MARGIN - 2 * _ROW + 3
        x, limit = frame.x, painter.width - _MARGIN
        hint = painter.elide(_VIEW_HINT, painter.width - 2 * _MARGIN, size=size, font_family=family)
        write(hint, _MARGIN, painter.height - _MARGIN - _ROW + 3, size, color)
        items: tuple[PointSeries3D | PlaneSeries, ...] = (*self._volume, *(() if self._plane is None else (self._plane,)))
        for item in items:
            if x > limit:
                break
            if isinstance(item, PointSeries3D):
                _marker(painter, x + 8, y + 8, item.shape, item.color, _VOLUME_MARKER + 2)
            else:
                painter.line(x, y + 8, x + 16, y + 8, item.color, 2)
            title = painter.elide(item.title, max(0, limit - x - 22), size=size, font_family=family)
            write(title, x + 22, y, size, color)
            x += 22 + text_width(title, size) + 24


def _check_color(color: str) -> None:
    """Reject a color the painter cannot use."""
    if not _COLOR.fullmatch(color):
        raise ValueError(f"Expected a #RRGGBB or #RRGGBBAA color, got {color!r}.")


def _validate_volume(points: tuple[PointSeries3D, ...], plane: PlaneSeries | None) -> None:
    """Reject a three-dimensional fit the chart cannot draw, before replacing the current one."""
    for series in points:
        if series.shape not in ("circle", "square"):
            raise ValueError(f"Expected a circle or square marker, got {series.shape!r}.")
        _check_color(series.color)
        for point in series.points:
            _check_point(point)
    if plane is None:
        return
    _check_color(plane.color)
    if len(plane.corners) != 4:
        raise ValueError("The regression plane needs four corners.")
    for corner in plane.corners:
        _check_point(corner)


def _check_point(point: object) -> None:
    """Reject a coordinate that is not three finite numbers."""
    if not isinstance(point, tuple) or len(point) != 3:
        raise ValueError("Chart coordinates must be finite numbers.")
    try:
        finite = all(isfinite(value) for value in point)
    except TypeError:
        finite = False
    if not finite:
        raise ValueError("Chart coordinates must be finite numbers.")


def _marker(painter: Painter, x: float, y: float, shape: Literal["circle", "square"], color: str, size: float) -> None:
    """Draw a marker of ``size`` pixels centred on (x, y)."""
    box = Rect(x - size / 2, y - size / 2, size, size)
    painter.rect(box, color, radius=size / 2 if shape == "circle" else 0)


def _thin[T](points: tuple[T, ...], limit: int = MAX_POINTS) -> tuple[T, ...]:
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


def _volume_extent(
    points: tuple[PointSeries3D, ...], plane: PlaneSeries | None
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return each axis range, widened so that no marker sits on the border of the box."""
    coordinates = [point for series in points for point in series.points]
    if plane is not None:
        coordinates.extend(plane.corners)
    if not coordinates:
        return (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    columns = list(zip(*coordinates, strict=True))
    return (
        _widen(min(columns[0]), max(columns[0])),
        _widen(min(columns[1]), max(columns[1])),
        _widen(min(columns[2]), max(columns[2])),
    )


def _frame(width: float, height: float) -> Rect | None:
    """Return the plot area above the legend, or ``None`` when the control is too small to draw it."""
    bottom = height - _MARGIN - 2 * _ROW
    if bottom - _MARGIN < _ROW or width - 2 * _MARGIN < _ROW:
        return None
    return Rect(_MARGIN, _MARGIN, width - 2 * _MARGIN, bottom - _MARGIN)


def _projector(
    spans: tuple[tuple[float, float], ...],
    azimuth: float,
    elevation: float,
    frame: Rect,
    zoom: float,
    pan_x: float,
    pan_y: float,
) -> Callable[[Point3], _Projected]:
    """Return a function from data coordinates to screen x, screen y and depth."""
    cube = [_view(x, y, z, azimuth, elevation) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    horizontals = [point[0] for point in cube]
    verticals = [point[1] for point in cube]
    min_h, max_h = min(horizontals), max(horizontals)
    min_v, max_v = min(verticals), max(verticals)
    fitted = _FIT * min(frame.width / max(max_h - min_h, 1e-6), frame.height / max(max_v - min_v, 1e-6))
    scale = fitted * zoom if zoom > 0 else fitted
    origin_x, origin_y = frame.x + frame.width / 2 + pan_x, frame.y + frame.height / 2 + pan_y
    center_h, center_v = (min_h + max_h) / 2, (min_v + max_v) / 2

    def project(point: Point3) -> _Projected:
        unit = tuple(_unit(point[index], spans[index][0], spans[index][1]) for index in range(3))
        horizontal, vertical, depth = _view(unit[0], unit[1], unit[2], azimuth, elevation)
        return origin_x + (horizontal - center_h) * scale, origin_y - (vertical - center_v) * scale, depth

    return project


def _limit_pan(x: float, y: float, width: float, height: float, zoom: float) -> tuple[float, float]:
    """Keep a pan inside the chart. An unknown size leaves the pan unchanged."""
    if width <= 1 or height <= 1:
        return x, y
    reach_x, reach_y = width * max(zoom, 1.0), height * max(zoom, 1.0)
    return min(reach_x, max(-reach_x, x)), min(reach_y, max(-reach_y, y))


def _view(x: float, y: float, z: float, azimuth: float, elevation: float) -> _Projected:
    """Map a point of the unit cube to horizontal, vertical and depth.

    Depth grows toward the camera. With the default angles, the first input goes right, the second goes left and the
    output goes up.
    """
    turn, lift = radians(azimuth), radians(elevation)
    horizontal = x * cos(turn) + y * sin(turn)
    toward = -x * sin(turn) + y * cos(turn)
    return horizontal, z * cos(lift) - toward * sin(lift), z * sin(lift) + toward * cos(lift)


def _unit(value: float, low: float, high: float) -> float:
    """Scale ``value`` into the unit interval between the ends of an axis."""
    span = high - low
    return 0.5 if span == 0 else (value - low) / span


def _box_edges(spans: tuple[tuple[float, float], ...]) -> tuple[tuple[Point3, Point3], ...]:
    """Return the twelve edges of the axis box."""
    corners = [(x, y, z) for x in spans[0] for y in spans[1] for z in spans[2]]
    edges = []
    for index, start in enumerate(corners):
        for end in corners[index + 1 :]:
            if sum(start[axis] != end[axis] for axis in range(3)) == 1:
                edges.append((start, end))
    return tuple(edges)


def _front_edges(
    edges: tuple[tuple[Point3, Point3], ...], project: Callable[[Point3], _Projected]
) -> tuple[tuple[Point3, Point3], tuple[Point3, Point3], tuple[Point3, Point3]]:
    """Pick the edge closest to the camera for each axis, which is where the ticks stay readable."""
    chosen = []
    for axis in range(3):
        parallel = [edge for edge in edges if edge[0][axis] != edge[1][axis]]
        chosen.append(max(parallel, key=lambda edge: project(_midpoint_edge(edge))[2]))
    return chosen[0], chosen[1], chosen[2]


def _plane_polylines(corners: tuple[Point3, ...], steps: int) -> tuple[tuple[Point3, ...], ...]:
    """Return the grid lines of the plane.

    ``corners`` follow ``(u = 0, v = 0)``, ``(1, 0)``, ``(1, 1)``, ``(0, 1)``. A bilinear grid lies on the plane.
    """

    def at(u: float, v: float) -> Point3:
        return _lerp(_lerp(corners[0], corners[1], u), _lerp(corners[3], corners[2], u), v)

    lines = []
    for step in range(steps + 1):
        fixed = step / steps
        lines.append(tuple(at(fixed, index / steps) for index in range(steps + 1)))
        lines.append(tuple(at(index / steps, fixed) for index in range(steps + 1)))
    return tuple(lines)


def _draw_edge(painter: Painter, edge: tuple[Point3, Point3], project: Callable[[Point3], _Projected], color: str, width: float) -> None:
    """Draw one edge of the axis box."""
    start, end = project(edge[0]), project(edge[1])
    painter.line(start[0], start[1], end[0], end[1], color, width)


def _lerp(start: Point3, end: Point3, t: float) -> Point3:
    """Return the point ``t`` of the way from ``start`` to ``end``."""
    return (
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
        start[2] + (end[2] - start[2]) * t,
    )


def _along(start: Point3, end: Point3, axis: int, value: float) -> Point3:
    """Return the point of an axis edge whose ``axis`` coordinate is ``value``."""
    span = end[axis] - start[axis]
    t = 0.0 if span == 0 else (value - start[axis]) / span
    return _lerp(start, end, t)


def _midpoint(spans: tuple[tuple[float, float], ...]) -> Point3:
    """Return the center of the axis box."""
    return (spans[0][0] + spans[0][1]) / 2, (spans[1][0] + spans[1][1]) / 2, (spans[2][0] + spans[2][1]) / 2


def _midpoint_edge(edge: tuple[Point3, Point3]) -> Point3:
    """Return the midpoint of an edge."""
    return _lerp(edge[0], edge[1], 0.5)


def _series_depth(points: tuple[Point3, ...], project: Callable[[Point3], _Projected]) -> float:
    """Return how close a series is, so a nearer series is painted later. An empty series stays at the back."""
    if not points:
        return -inf
    return min(project(point)[2] for point in points)


def _outward(x: float, y: float, center: _Projected, distance: float) -> tuple[float, float]:
    """Step from a point away from the center of the box, so labels sit outside it."""
    dx, dy = x - center[0], y - center[1]
    length = hypot(dx, dy)
    if length < 1:
        dx, dy, length = 0.0, -1.0, 1.0
    return x + dx / length * distance, y + dy / length * distance


def _place_text(
    write: Callable[[str, float, float, float, str], None],
    text: str,
    anchor: tuple[float, float],
    width: float,
    center: _Projected,
    painter: Painter,
    size: float,
    color: str,
) -> None:
    """Draw ``text`` beside ``anchor``, on the side that faces away from the box, and keep it inside the control."""
    x, y = anchor
    if x < center[0]:
        x -= width
    y -= size * 0.7
    x = min(max(0, x), max(0, painter.width - width))
    y = min(max(0, y), max(0, painter.height - size))
    write(text, x, y, size, color)
