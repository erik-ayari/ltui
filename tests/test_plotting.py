from lightning_tui.plotting import PlotCurve, prepare_curve


def test_log_scaling_drops_nonpositive_values() -> None:
    curve = PlotCurve(label="loss", x=(-1.0, 0.0, 1.0, 10.0), y=(1.0, 2.0, 3.0, 4.0))

    scaled, dropped_x, dropped_y = prepare_curve(curve, smoothing=False, log_x=True, log_y=False)

    assert scaled.x == (0.0, 1.0)
    assert scaled.y == (3.0, 4.0)
    assert dropped_x == 2
    assert dropped_y == 0
