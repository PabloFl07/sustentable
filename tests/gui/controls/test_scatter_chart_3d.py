"""Tests for the three-dimensional fit chart: its plane, its axes and dragging to rotate it."""

import math

import pytest
from pysual import Painter, Rect
from pysual.host import Input

from sustentable.gui.controls.scatter_chart import (
    DEFAULT_AZIMUTH,
    DEFAULT_ELEVATION,
    MAX_POINTS,
    PLANE_STEPS,
    PlaneSeries,
    PointSeries3D,
    ScatterChart,
)

TRAIN = "#60a5fa"
TEST = "#fb923c"
PLANE = "#34d399"
CORNERS = ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 2.0), (0.0, 1.0, 1.0))


class _RecordingHost:
    """Record what a control draws, measuring text as a fixed width per character."""

    capabilities = frozenset()
    size = (640.0, 400.0)

    def __init__(self) -> None:
        """Start with no recorded drawing."""
        self.rects: list[tuple[Rect, str, float]] = []
        self.texts: list[str] = []
        self.lines: list[tuple[float, float, float, float, str]] = []

    def clip(self, rect: Rect | None) -> None:
        """Ignore clipping, which these tests do not check."""

    def rect(self, rect: Rect, fill: str, radius: float = 0, border: str = "", border_width: float = 1) -> None:
        """Record a filled rectangle; markers are drawn this way."""
        self.rects.append((rect, fill, radius))

    def text(self, text: str, x: float, y: float, color: str, size: float, mono: bool = False) -> None:
        """Record the drawn text."""
        self.texts.append(text)

    def measure(self, text: str, size: float, mono: bool = False) -> tuple[float, float]:
        """Give every character the same width."""
        return len(text) * size * 0.6, size * 1.3

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1) -> None:
        """Record a straight line."""
        self.lines.append((x1, y1, x2, y2, color))

    def markers(self, color: str) -> list[tuple[Rect, float]]:
        """Rectangles drawn with ``color``, in drawing order, with their corner radius."""
        return [(rect, radius) for rect, fill, radius in self.rects if fill == color]

    def numbers(self) -> list[float]:
        """Every drawn text that is a number, such as the tick labels."""
        values = []
        for text in self.texts:
            try:
                values.append(float(text.replace(",", "")))
            except ValueError:
                pass
        return values


def _paint(chart: ScatterChart) -> _RecordingHost:
    """Paint ``chart`` on a 640 x 400 recording host."""
    host = _RecordingHost()
    chart.paint(Painter(host, Rect(0, 0, *host.size), chart.effective_theme, chart))
    return host


def _chart() -> ScatterChart:
    """A chart with both subsets and a plane through the training points."""
    chart = ScatterChart(x_label="a", y_label="b", z_label="y")
    chart.set_volume(
        [PointSeries3D("Training", ((0, 0, 0), (1, 0, 1)), TRAIN), PointSeries3D("Test", ((0, 1, 2),), TEST, "square")],
        PlaneSeries("Regression plane", CORNERS, PLANE),
    )
    return chart


def _centers(host: _RecordingHost, color: str) -> list[tuple[float, float]]:
    """Centers of the data markers, leaving out the larger one drawn in the legend."""
    sizes = {rect.width for rect, _ in host.markers(color)}
    data = [rect for rect, _ in host.markers(color) if rect.width == min(sizes)]
    return [(rect.x + rect.width / 2, rect.y + rect.height / 2) for rect in data]


def test_points_plane_axis_titles_and_legend_are_painted() -> None:
    """Each subset has its own marker, and the legend names the series, the plane and the drag hint."""
    host = _paint(_chart())

    circles, squares = host.markers(TRAIN), host.markers(TEST)
    assert len(circles) == 2 + 1 and all(radius > 0 for _, radius in circles)
    assert len(squares) == 1 + 1 and all(radius == 0 for _, radius in squares)
    assert {"a", "b", "y", "Training", "Test", "Regression plane", "Drag to rotate · Shift-drag to pan · Scroll to zoom"} <= set(host.texts)
    grid = 2 * (PLANE_STEPS + 1) * PLANE_STEPS
    assert sum(color == PLANE for *_, color in host.lines) == grid + 1, "Every grid segment, plus the legend sample."
    xs = [end for x1, y1, x2, y2, color in host.lines if color == PLANE for end in (x1, x2)]
    ys = [end for x1, y1, x2, y2, color in host.lines if color == PLANE for end in (y1, y2)]
    assert max(xs) - min(xs) > 30 and max(ys) - min(ys) > 30


def test_default_view_sends_each_input_a_different_way_and_the_output_up() -> None:
    """The first input goes right, the second goes left and a larger output is higher on the screen."""
    chart = ScatterChart()
    chart.set_volume([PointSeries3D("Training", ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), TRAIN)])

    centers = _centers(_paint(chart), TRAIN)
    assert len(centers) == 4
    top = min(centers, key=lambda point: point[1])
    right = max(centers, key=lambda point: point[0])
    left = min(centers, key=lambda point: point[0])
    assert top is not left and top is not right and left is not right
    origin = next(point for point in centers if point is not top and point is not left and point is not right)
    assert top[1] < origin[1]
    assert left[0] < origin[0] < right[0]


def test_dragging_rotates_the_view_and_stops_before_it_flips() -> None:
    """Dragging sideways turns the chart; a long vertical drag keeps the output axis pointing up."""
    chart = _chart()
    before = _paint(chart).lines
    chart.handle_input(Input(kind="pointer_down", x=10, y=20, button=1, pointer_id=3))
    chart.handle_input(Input(kind="pointer_move", x=110, y=20, button=1, pointer_id=3))

    assert chart.azimuth == pytest.approx(DEFAULT_AZIMUTH + 40)
    assert chart.elevation == DEFAULT_ELEVATION
    assert _paint(chart).lines != before

    chart.handle_input(Input(kind="pointer_move", x=110, y=-10_000, button=1, pointer_id=3))
    assert chart.elevation == 80
    chart.handle_input(Input(kind="pointer_move", x=110, y=10_000, button=1, pointer_id=3))
    assert chart.elevation == 12
    chart.handle_input(Input(kind="pointer_up", x=110, y=10_000, button=1, pointer_id=3))
    chart.handle_input(Input(kind="pointer_move", x=400, y=10_000, button=1, pointer_id=3))
    assert chart.azimuth == pytest.approx(DEFAULT_AZIMUTH + 40)
    assert chart.elevation == 12


def _plane_width(host: _RecordingHost) -> float:
    """Width of the drawn regression plane, ignoring the short sample in the legend."""
    xs = [end for x1, _, x2, _, color in host.lines if color == PLANE for end in (x1, x2)]
    return max(xs) - min(xs)


def test_the_wheel_zooms_and_stops_at_the_limits() -> None:
    """Scrolling up enlarges the plane; scrolling down shrinks it. Either direction stops at a limit."""
    chart = _chart()
    width = _plane_width(_paint(chart))
    assert chart.handle_input(Input(kind="wheel", delta=-1)) is True

    assert chart.zoom == pytest.approx(1.2)
    assert _plane_width(_paint(chart)) > width
    chart.handle_input(Input(kind="wheel", delta=1))
    assert chart.zoom == pytest.approx(1)
    for _ in range(30):
        chart.handle_input(Input(kind="wheel", delta=-1))
    assert chart.zoom == 5
    for _ in range(30):
        chart.handle_input(Input(kind="wheel", delta=1))
    assert chart.zoom == 0.5


def test_shift_drag_pans_without_rotating_and_stops_at_the_chart_edge() -> None:
    """Holding Shift moves the picture with the pointer and leaves the camera angle alone."""
    chart = _chart()
    centers = _centers(_paint(chart), TRAIN)
    chart.handle_input(Input(kind="pointer_down", x=10, y=30, button=1, shift=True, pointer_id=4))
    chart.handle_input(Input(kind="pointer_move", x=50, y=15, button=1, pointer_id=4))

    assert (chart.pan_x, chart.pan_y) == (40, -15)
    assert (chart.azimuth, chart.elevation) == (DEFAULT_AZIMUTH, DEFAULT_ELEVATION)
    moved = _centers(_paint(chart), TRAIN)
    assert moved[0][0] == pytest.approx(centers[0][0] + 40)
    assert moved[0][1] == pytest.approx(centers[0][1] - 15)

    chart._rect = Rect(0, 0, 200, 120)
    chart.handle_input(Input(kind="pointer_move", x=10_000, y=-10_000, button=1, pointer_id=4))
    assert (chart.pan_x, chart.pan_y) == (200, -120)


def test_a_different_pointer_does_not_rotate_the_view() -> None:
    """A move from another pointer leaves the drag that is already in progress alone."""
    chart = _chart()
    chart.handle_input(Input(kind="pointer_down", x=0, y=0, button=1, pointer_id=1))
    chart.handle_input(Input(kind="pointer_move", x=80, y=0, button=1, pointer_id=2))

    assert chart.azimuth == DEFAULT_AZIMUTH and chart.elevation == DEFAULT_ELEVATION


def test_new_data_returns_to_the_default_view() -> None:
    """Replacing the series looks at them from the front again."""
    chart = _chart()
    chart.handle_input(Input(kind="pointer_down", x=0, y=0, button=1, pointer_id=1))
    chart.handle_input(Input(kind="pointer_move", x=50, y=0, button=1, pointer_id=1))
    chart.handle_input(Input(kind="pointer_up", x=50, y=0, button=1, pointer_id=1))
    chart.handle_input(Input(kind="wheel", delta=-1))
    chart.handle_input(Input(kind="pointer_down", x=0, y=0, button=1, shift=True, pointer_id=1))
    chart.handle_input(Input(kind="pointer_move", x=20, y=10, button=1, pointer_id=1))

    chart.set_volume([PointSeries3D("Training", ((0, 0, 0), (1, 1, 1)), TRAIN)])

    assert (chart.azimuth, chart.elevation, chart.zoom) == (DEFAULT_AZIMUTH, DEFAULT_ELEVATION, 1)
    assert (chart.pan_x, chart.pan_y) == (0, 0)


def test_long_series_are_thinned_but_the_axes_cover_every_point() -> None:
    """At most ``MAX_POINTS`` markers per series are painted, starting with the first point."""
    points = tuple((float(index), 0.0, float(index)) for index in range(3 * MAX_POINTS + 1))
    chart = ScatterChart()
    chart.set_volume([PointSeries3D("Training", points, TRAIN)])

    (drawn,) = chart.drawn_points
    assert len(drawn) == MAX_POINTS and drawn[0] == points[0]
    assert chart.point_series[0].points is points
    assert max(_paint(chart).numbers()) >= points[-1][0]


def test_a_single_point_gets_a_range_around_it() -> None:
    """Equal values do not collapse an axis."""
    chart = ScatterChart()
    chart.set_volume([PointSeries3D("Training", ((5, 5, 5),), TRAIN)])

    host = _paint(chart)

    assert len(host.markers(TRAIN)) == 2
    assert min(host.numbers()) < 5 < max(host.numbers())


def test_invalid_data_keeps_the_previous_series_and_the_view() -> None:
    """Non-finite coordinates, a short plane and unknown colors are rejected without changing the chart."""
    chart = _chart()
    chart.handle_input(Input(kind="pointer_down", x=0, y=0, button=1, pointer_id=1))
    chart.handle_input(Input(kind="pointer_move", x=30, y=0, button=1, pointer_id=1))
    chart.handle_input(Input(kind="wheel", delta=-1))
    points, plane, azimuth, zoom = chart.point_series, chart.plane, chart.azimuth, chart.zoom

    with pytest.raises(ValueError, match="finite"):
        chart.set_volume([PointSeries3D("Training", ((0, 0, math.nan),), TRAIN)])
    with pytest.raises(ValueError, match="four corners"):
        chart.set_volume([], PlaneSeries("Plane", CORNERS[:3], PLANE))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="color"):
        chart.set_volume([], PlaneSeries("Plane", CORNERS, "red"))
    with pytest.raises(ValueError, match="marker"):
        chart.set_volume([PointSeries3D("Training", ((0, 0, 0),), TRAIN, "triangle")])  # type: ignore[arg-type]

    assert chart.point_series is points and chart.plane is plane and chart.azimuth == azimuth and chart.zoom == zoom


def test_empty_chart_shows_its_message() -> None:
    """Without data the chart only writes ``empty_text``; ``clear`` returns to that state and the default view."""
    chart = _chart()
    chart.handle_input(Input(kind="pointer_down", x=0, y=0, button=1, pointer_id=1))
    chart.handle_input(Input(kind="pointer_move", x=40, y=0, button=1, pointer_id=1))
    chart.handle_input(Input(kind="wheel", delta=-1))
    chart.clear()
    chart.empty_text = "Create the model to see its fit."

    host = _paint(chart)

    assert not chart.point_series and chart.plane is None
    assert (chart.azimuth, chart.elevation, chart.zoom) == (DEFAULT_AZIMUTH, DEFAULT_ELEVATION, 1)
    assert (chart.pan_x, chart.pan_y) == (0, 0)
    assert host.texts == ["Create the model to see its fit."]
    assert not host.lines
