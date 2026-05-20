from unittest.mock import patch

import pytest

from lightning_tui.plotting import PlotCurve, plot_val, prepare_curve, render_plot


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


def test_val_curve_uses_connected_dotted_line() -> None:
    curve = PlotCurve(label="val", x=(1.0, 10.0, 20.0), y=(0.9, 0.7, 0.6), role="val")

    with patch("lightning_tui.plotting.plt.plot") as plot:
        plot_val(curve)

    plot.assert_called_once_with(curve.x, curve.y, label="val", color="blue", marker="dot")
