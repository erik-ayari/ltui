from unittest.mock import patch

import pytest

from lightning_tui.plotting import PlotBounds, PlotCurve, dashed_segments, prepare_curve, render_plot


def test_log_scaling_drops_nonpositive_values() -> None:
    curve = PlotCurve(label="loss", x=(-1.0, 0.0, 1.0, 10.0), y=(1.0, 2.0, 3.0, 4.0))

    scaled, dropped_x, dropped_y = prepare_curve(curve, smoothing=False, log_x=True, log_y=False)

    assert scaled.x == (0.0, 1.0)
    assert scaled.y == (3.0, 4.0)
    assert dropped_x == 2
    assert dropped_y == 0


def test_render_plot_can_force_x_axis_to_zero() -> None:
    curve = PlotCurve(label="train", x=(24.0, 49.0), y=(1.0, 0.8))

    with patch("lightning_tui.plotting.plt.xlim") as xlim, patch("lightning_tui.plotting.plt.ylim") as ylim:
        render_plot([curve], width=60, height=20, x_min=0)

    assert xlim.call_args.kwargs["left"] == 0
    assert xlim.call_args.kwargs["right"] == pytest.approx(50.47)
    ylim.assert_called_once()


def test_dashed_segments_use_screen_columns() -> None:
    curve = PlotCurve(label="val", x=tuple(float(value) for value in range(32)), y=tuple(1.0 for value in range(32)))
    bounds = PlotBounds(x_left=0, x_right=31, y_lower=0, y_upper=2)

    segments = dashed_segments(curve, bounds, width=32)

    assert len(segments) > 1
    assert all(len(x_segment) <= 4 for x_segment, y_segment in segments)
