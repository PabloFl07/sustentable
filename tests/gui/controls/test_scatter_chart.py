"""Tests for the scatter chart's data handling and painting, using a host that records drawing calls."""

import math

import pytest
from pysual import Painter, Rect

from sustentable.gui.controls.scatter_chart import MAX_POINTS, LineSeries, PointSeries, ScatterChart

TRAIN = "#60a5fa"
TEST = "#fb923c"
LINE = "#34d399"


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


def _fit_chart() -> ScatterChart:
    """A chart with training circles, test squares and a regression line through the training points."""
    chart = ScatterChart(x_label="area", y_label="energy")
    chart.set_data(
        [PointSeries("Training", ((0, 20), (5, 25), (9, 29)), TRAIN), PointSeries("Test", ((2, 23), (7, 26)), TEST, "square")],
        [LineSeries("Regression line", (0, 20), (9, 29), LINE)],
    )
    return chart


def test_points_line_axis_titles_and_legend_are_painted() -> None:
    """Each subset has its own marker shape and color, and the legend and axis titles name everything drawn."""
    host = _paint(_fit_chart())

    circles, squares = host.markers(TRAIN), host.markers(TEST)
    # One marker per point, plus the one in the legend.
    assert len(circles) == 3 + 1 and all(radius > 0 for _, radius in circles)
    assert len(squares) == 2 + 1 and all(radius == 0 for _, radius in squares)
    assert {"area", "energy", "Training", "Test", "Regression line"} <= set(host.texts)
    assert sum(color == LINE for *_, color in host.lines) == 2, "The fitted line and its legend sample."


def test_markers_and_line_share_the_same_axes() -> None:
    """Larger values are further right and higher, and the line starts and ends on the points it passes through."""
    host = _paint(_fit_chart())

    centers = [(rect.x + rect.width / 2, rect.y + rect.height / 2) for rect, _ in host.markers(TRAIN)[:3]]
    assert centers[0][0] < centers[1][0] < centers[2][0]
    assert centers[0][1] > centers[1][1] > centers[2][1]
    x1, y1, x2, y2, _ = next(line for line in host.lines if line[4] == LINE)
    assert (x1, y1) == pytest.approx(centers[0])
    assert (x2, y2) == pytest.approx(centers[2])


@pytest.mark.parametrize(
    ("points", "labels"),
    [
        (((0, 20), (9, 29)), {"0", "2", "4", "6", "8", "20", "24", "28"}),
        (((0, 0), (1, 0.5)), {"0.0", "0.1", "0.3", "0.5", "1.0"}),
        (((0, 100_000), (1, 500_000)), {"100,000", "300,000", "500,000"}),
    ],
    ids=["units", "decimals", "thousands"],
)
def test_tick_labels_are_round_numbers(points: tuple[tuple[float, float], ...], labels: set[str]) -> None:
    """Ticks are steps of 1, 2 or 5 times a power of ten, with the decimals they need and thousands separators."""
    chart = ScatterChart()
    chart.set_data([PointSeries("Training", points, TRAIN)])

    assert labels <= set(_paint(chart).texts)


def test_long_series_are_thinned_but_the_axes_cover_every_point() -> None:
    """At most ``MAX_POINTS`` markers per series are painted, starting with the first point."""
    points = tuple((float(index), float(index)) for index in range(3 * MAX_POINTS + 1))
    chart = ScatterChart()
    chart.set_data([PointSeries("Training", points, TRAIN)])

    (drawn,) = chart.drawn_points
    assert len(drawn) == MAX_POINTS and drawn[0] == points[0]
    assert chart.point_series[0].points is points
    host = _paint(chart)
    assert len(host.markers(TRAIN)) == MAX_POINTS + 1
    assert max(host.numbers()) >= points[-1][0]


def test_a_single_point_gets_a_range_around_it() -> None:
    """Equal values do not collapse an axis."""
    chart = ScatterChart()
    chart.set_data([PointSeries("Training", ((5, 5),), TRAIN)])

    host = _paint(chart)

    assert len(host.markers(TRAIN)) == 2
    assert min(host.numbers()) < 5 < max(host.numbers())


def test_invalid_data_keeps_the_previous_series() -> None:
    """Non-finite coordinates and unknown colors are rejected without changing the chart."""
    chart = _fit_chart()
    points, lines = chart.point_series, chart.line_series

    with pytest.raises(ValueError, match="finite"):
        chart.set_data([PointSeries("Training", ((0, math.nan),), TRAIN)])
    with pytest.raises(ValueError, match="color"):
        chart.set_data([], [LineSeries("Line", (0, 0), (1, 1), "red")])

    assert chart.point_series is points and chart.line_series is lines


def test_empty_chart_shows_its_message() -> None:
    """Without data the chart only writes ``empty_text``; ``clear`` returns to that state."""
    chart = _fit_chart()
    chart.clear()
    chart.empty_text = "Create the model to see its fit."

    host = _paint(chart)

    assert not chart.point_series and not chart.line_series
    assert host.texts == ["Create the model to see its fit."]
    assert not host.lines
