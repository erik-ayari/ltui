from unittest.mock import patch

import pytest

from ltui.plotting import (
    LegendEntry,
    PlotBounds,
    PlotCurve,
    PlotPanel,
    color_active_run_labels,
    draw_legend_entries,
    format_log_tick,
    grid_layout,
    integer_ticks,
    log_ticks,
    plot_val,
    prepare_curve,
    render_plot_grid,
    render_plot,
)


def test_log_scaling_drops_nonpositive_values() -> None:
    curve = PlotCurve(label="loss", x=(-1.0, 0.0, 1.0, 10.0), y=(1.0, 2.0, 3.0, 4.0))

    scaled, dropped_x, dropped_y = prepare_curve(curve, smoothing=False, log_x=True, log_y=False)

    assert scaled.x == (0.0, 1.0)
    assert scaled.y == (3.0, 4.0)
    assert dropped_x == 2
    assert dropped_y == 0


def test_render_plot_can_force_x_axis_to_zero() -> None:
    curve = PlotCurve(label="train", x=(24.0, 49.0), y=(1.0, 0.8))

    with patch("ltui.plotting.plt.xlim") as xlim, patch("ltui.plotting.plt.ylim") as ylim:
        render_plot([curve], width=60, height=20, x_min=0)

    assert xlim.call_args.kwargs["left"] == 0
    assert xlim.call_args.kwargs["right"] == pytest.approx(50.47)
    ylim.assert_called_once()


def test_render_plot_uses_integer_step_ticks() -> None:
    curve = PlotCurve(label="train", x=(24.0, 49.0), y=(1.0, 0.8))

    with patch("ltui.plotting.plt.xticks") as xticks:
        render_plot([curve], width=60, height=20, x_label="step", x_min=0)

    ticks, labels = xticks.call_args.args
    assert ticks == [0, 10, 20, 30, 40, 50]
    assert labels == ["0", "10", "20", "30", "40", "50"]


def test_render_plot_uses_power_labels_for_log_y() -> None:
    curve = PlotCurve(label="loss", x=(0.0, 1.0), y=(0.1, 10.0))

    with patch("ltui.plotting.plt.yticks") as yticks:
        render_plot([curve], width=60, height=20, log_y=True)

    ticks, labels = yticks.call_args.args
    assert ticks == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert labels == ["10⁻¹", "10⁻⁰·⁵", "10⁰", "10⁰·⁵", "10¹"]


def test_log_ticks_format_exponents_as_powers() -> None:
    ticks, labels = log_ticks(-2.1, 1.1, target_count=5)

    assert ticks == [-2, -1, 0, 1]
    assert labels == [format_log_tick(tick) for tick in ticks]


def test_integer_ticks_are_discrete_for_small_epoch_range() -> None:
    assert integer_ticks(0, 3.2) == [0, 1, 2, 3]


def test_render_plot_sets_title_and_terminal_dark_theme() -> None:
    curve = PlotCurve(label="train", x=(0.0, 1.0), y=(1.0, 0.8))

    with patch("ltui.plotting.plt.title") as title, patch("ltui.plotting.plt.theme") as theme, patch("ltui.plotting.plt.ticks_color") as ticks_color:
        render_plot([curve], width=60, height=20, title="loss", dark_mode=True)

    title.assert_called_once_with("loss")
    theme.assert_called_once_with("clear")
    ticks_color.assert_called_once_with("green")


def test_render_plot_can_use_light_theme() -> None:
    curve = PlotCurve(label="train", x=(0.0, 1.0), y=(1.0, 0.8))

    with patch("ltui.plotting.plt.theme") as theme:
        render_plot([curve], width=60, height=20, dark_mode=False)

    theme.assert_called_once_with("default")


def test_val_curve_uses_connected_dotted_line() -> None:
    curve = PlotCurve(label="val", x=(1.0, 10.0, 20.0), y=(0.9, 0.7, 0.6), role="val")

    with patch("ltui.plotting.plt.plot") as plot:
        plot_val(curve)

    plot.assert_called_once_with(curve.x, curve.y, label=None, color="blue", marker="dot")


def test_custom_legend_entries_use_run_colors_then_neutral_styles() -> None:
    curves = [
        PlotCurve(
            "run_a train",
            (0.0,),
            (1.0,),
            color="blue",
            role="train",
            run_label="run_a",
            run_status="active",
            style_label="train",
        ),
        PlotCurve("run_a val", (0.0,), (0.8,), color="blue", role="val", run_label="run_a", run_status="active", style_label="val"),
        PlotCurve("run_b train", (0.0,), (0.9,), color="red", role="train", run_label="run_b", run_status="stale", style_label="train"),
    ]
    bounds = PlotBounds(x_left=0, x_right=10, y_lower=0, y_upper=1)

    with patch("ltui.plotting.plt.plot") as plot:
        entries = draw_legend_entries(curves, bounds, dark_mode=True)

    labels = [call.kwargs["label"] for call in plot.call_args_list]
    colors = [call.kwargs["color"] for call in plot.call_args_list]
    markers = [call.kwargs["marker"] for call in plot.call_args_list]
    assert labels == ["run_a", "run_b", "train", "val"]
    assert colors == ["blue", "red", "white", "white"]
    assert markers == ["dot", "dot", "braille", "dot"]
    assert entries == [LegendEntry("run_a", "blue", True), LegendEntry("run_b", "red", False)]


def test_active_run_labels_are_colored_green_after_render() -> None:
    text = "run_a run_b run_a"

    colored = color_active_run_labels(text, [LegendEntry("run_a", "blue", True), LegendEntry("run_b", "red", False)], "\033[38;5;3m")

    assert colored == "\033[38;5;10mrun_a\033[38;5;3m run_b run_a"


def test_single_run_still_gets_run_legend_entry() -> None:
    curves = [
        PlotCurve(
            "train",
            (0.0,),
            (1.0,),
            color="blue",
            role="train",
            run_label="stage3/version_0",
            run_status="active",
            style_label="train",
        )
    ]
    bounds = PlotBounds(x_left=0, x_right=10, y_lower=0, y_upper=1)

    with patch("ltui.plotting.plt.plot") as plot:
        entries = draw_legend_entries(curves, bounds, dark_mode=True)

    assert entries == [LegendEntry("stage3/version_0", "blue", True)]
    assert plot.call_args_list[0].kwargs["label"] == "stage3/version_0"


def test_grid_layout_pages_when_terminal_is_too_small() -> None:
    layout = grid_layout(total=5, width=80, height=20)

    assert layout.page_size < 5
    assert layout.page_count > 1


def test_render_plot_grid_marks_selected_panel() -> None:
    panel = PlotPanel(
        title="loss",
        curves=[PlotCurve(label="train", x=(0.0, 1.0), y=(1.0, 0.8))],
        x_label="step",
        y_label="loss",
    )

    result = render_plot_grid([panel], width=80, height=20, selected_index=0)

    assert "┌" in result.text
    assert "┘" in result.text


def test_render_plot_grid_shows_legend_only_in_top_left_panel() -> None:
    panels = [
        PlotPanel(
            title=f"metric_{index}",
            curves=[PlotCurve(label="train", x=(0.0, 1.0), y=(1.0, 0.8), role="train", style_label="train")],
            x_label="step",
        )
        for index in range(4)
    ]

    with patch("ltui.plotting.render_plot") as render:
        render.return_value.text = "plot"
        render.return_value.status_messages = ()
        render_plot_grid(panels, width=120, height=40)

    assert [call.kwargs["show_legend"] for call in render.call_args_list] == [True, False, False, False]
