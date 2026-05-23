import asyncio
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ltui.app import (
    PALETTE,
    ConfigScreen,
    LightningTuiApp,
    SelectorScreen,
    keybinding_bar,
)
from ltui.plotting import PlotResult


def write_multimetric_log(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "epoch,step,train_loss,val_loss,train_accuracy,val_accuracy,train_kl,val_kl\n"
        "0,0,1.0,,0.4,,0.2,\n"
        "0,1,0.8,0.7,0.5,0.45,0.15,0.18\n"
        "1,2,0.6,0.55,0.65,0.6,0.1,0.12\n"
    )


def test_metric_selector_opens_without_shadowing_textual_query() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "lightning_logs" / "version_0" / "metrics.csv"
            metrics.parent.mkdir(parents=True)
            metrics.write_text("step,train_loss,val_loss\n0,1.0,\n1,,0.8\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(100, 35)) as pilot:
                await pilot.pause(0.5)
                await pilot.press("m")
                await pilot.pause(0.1)

                assert isinstance(app.screen, SelectorScreen)

    asyncio.run(run())


def test_config_selector_opens_for_runs_with_unique_yaml() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "version_0" / "metrics.csv"
            config = root / "version_0" / "config.yaml"
            metrics.parent.mkdir(parents=True)
            metrics.write_text("step,train_loss\n0,1.0\n")
            config.write_text("trainer:\n  max_epochs: 2\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                await pilot.press("c")
                await pilot.pause(0.2)

                assert isinstance(app.screen, ConfigScreen)
                assert "trainer:" in app.screen.query_one("#config-preview").text
                assert "[ ]" not in str(app.screen.query_one("#config-options").get_option_at_index(0).prompt)

    asyncio.run(run())


def test_run_selector_groups_versions_by_model_and_model_selects_all() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "stage3" / "version_0" / "metrics.csv"
            second = root / "stage3" / "version_1" / "metrics.csv"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("step,train_loss\n0,1.0\n")
            second.write_text("step,train_loss\n0,0.8\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                app.selected_run_paths = [str(first.resolve())]
                await pilot.press("r")
                await pilot.pause(0.1)

                options = app.screen.query_one("#options")
                assert str(options.get_option_at_index(0).prompt).startswith("[-] stage3")
                assert "  v0" in str(options.get_option_at_index(1).prompt)

                await pilot.press("space")
                await pilot.press("enter")
                await pilot.pause(0.1)

                assert app.selected_run_paths == [str(first.resolve()), str(second.resolve())]

    asyncio.run(run())


def test_run_selector_left_right_jump_between_models() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                root / "stage3" / "version_0" / "metrics.csv",
                root / "stage3" / "version_1" / "metrics.csv",
                root / "stage4" / "mini" / "metrics.csv",
            ]
            for path in paths:
                path.parent.mkdir(parents=True)
                path.write_text("step,train_loss\n0,1.0\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                await pilot.press("r")
                await pilot.pause(0.1)
                options = app.screen.query_one("#options")

                await pilot.press("right")
                assert options.highlighted == 3
                await pilot.press("left")
                assert options.highlighted == 0

    asyncio.run(run())


def test_config_selector_groups_versions_by_model_and_model_preview_combines_configs() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "stage3" / "version_0" / "metrics.csv"
            second = root / "stage3" / "version_1" / "metrics.csv"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("step,train_loss\n0,1.0\n")
            second.write_text("step,train_loss\n0,0.8\n")
            (first.parent / "config.yaml").write_text("model: first\n")
            (second.parent / "config.yaml").write_text("model: second\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                await pilot.press("c")
                await pilot.pause(0.1)

                options = app.screen.query_one("#config-options")
                assert str(options.get_option_at_index(0).prompt) == "stage3"
                assert "  v0" in str(options.get_option_at_index(1).prompt)
                preview = app.screen.query_one("#config-preview").text
                assert "model: first" in preview
                assert "model: second" in preview

    asyncio.run(run())


def test_config_selector_left_right_jump_between_models() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                root / "stage3" / "version_0" / "metrics.csv",
                root / "stage3" / "version_1" / "metrics.csv",
                root / "stage4" / "mini" / "metrics.csv",
            ]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True)
                path.write_text("step,train_loss\n0,1.0\n")
                (path.parent / "config.yaml").write_text(f"model: {index}\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                await pilot.press("c")
                await pilot.pause(0.1)
                options = app.screen.query_one("#config-options")

                await pilot.press("right")
                assert options.highlighted == 3
                assert "model: 2" in app.screen.query_one("#config-preview").text
                await pilot.press("left")
                assert options.highlighted == 0
                assert "model: 0" in app.screen.query_one("#config-preview").text

    asyncio.run(run())


def test_axis_toggle_switches_step_epoch() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "lightning_logs" / "version_0" / "metrics.csv"
            metrics.parent.mkdir(parents=True)
            metrics.write_text("epoch,step,train_loss_step,train_loss_epoch,val_loss\n0,10,0.9,,\n0,20,,0.8,\n0,20,,,0.7\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(100, 35)) as pilot:
                await pilot.pause(0.5)

                assert app.x_axis_mode == "step"
                await pilot.press("a")
                await pilot.pause(0.1)

                assert app.x_axis_mode == "epoch"

    asyncio.run(run())


def test_dark_mode_toggle_switches_theme() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "lightning_logs" / "version_0" / "metrics.csv"
            metrics.parent.mkdir(parents=True)
            metrics.write_text("step,train_loss\n0,1.0\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(100, 35)) as pilot:
                await pilot.pause(0.5)

                assert app.dark_mode is True
                await pilot.press("d")
                await pilot.pause(0.1)

                assert app.dark_mode is False

    asyncio.run(run())


def test_grouped_legend_labels_use_train_val() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "lightning_logs" / "version_0" / "metrics.csv"
            metrics.parent.mkdir(parents=True)
            metrics.write_text("step,train_loss,val_loss\n0,1.0,\n1,,0.8\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(100, 35)) as pilot:
                await pilot.pause(0.5)

                assert [curve.label for curve in app.build_curves()] == ["train", "val"]

    asyncio.run(run())


def test_compare_labels_are_implicit_from_selected_runs() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "run_a" / "version_0" / "metrics.csv"
            second = root / "run_b" / "version_0" / "metrics.csv"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("step,train_loss\n0,1.0\n")
            second.write_text("step,train_loss\n0,0.8\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(100, 35)) as pilot:
                await pilot.pause(0.5)
                app.selected_run_paths = [str(first.resolve()), str(second.resolve())]
                app.selected_metrics = ["loss"]

                assert [curve.label for curve in app.build_curves()] == ["run_a/v0 train", "run_b/v0 train"]

    asyncio.run(run())


def test_multi_run_plotext_labels_include_runs_and_styles() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "run_a" / "version_0" / "metrics.csv"
            second = root / "run_b" / "version_0" / "metrics.csv"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("step,train_loss,val_loss\n0,1.0,\n1,,0.8\n")
            second.write_text("step,train_loss,val_loss\n0,0.9,\n1,,0.7\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                app.selected_run_paths = [str(first.resolve()), str(second.resolve())]
                app.selected_metrics = ["loss"]
                labels = [curve.label for curve in app.build_curves()]

                assert labels == [
                    "run_a/v0 train",
                    "run_a/v0 val",
                    "run_b/v0 train",
                    "run_b/v0 val",
                ]

    asyncio.run(run())


def test_four_selected_runs_use_distinct_early_colors() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_paths = []
            for index in range(4):
                metrics = root / f"run_{index}" / "version_0" / "metrics.csv"
                metrics.parent.mkdir(parents=True)
                metrics.write_text(f"step,train_loss\n0,{1 - index * 0.1}\n")
                metrics_paths.append(metrics)

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                app.selected_run_paths = [str(path.resolve()) for path in metrics_paths]
                app.selected_metrics = ["loss"]
                colors = [curve.color for curve in app.build_curves()]

                assert colors == list(PALETTE[:4])
                assert len(set(colors)) == 4

    asyncio.run(run())


def test_space_opens_multiplot_grid() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "lightning_logs" / "version_0" / "metrics.csv"
            write_multimetric_log(metrics)

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                app.selected_metrics = ["loss", "accuracy", "kl"]
                await pilot.press("space")
                await pilot.pause(0.1)

                assert app.multiplot is True
                assert app.multiplot_selection is None

    asyncio.run(run())


def test_multiplot_arrow_selects_and_enter_focuses_metric() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "lightning_logs" / "version_0" / "metrics.csv"
            write_multimetric_log(metrics)

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                app.selected_metrics = ["loss", "accuracy", "kl"]
                await pilot.press("space")
                await pilot.press("right")
                await pilot.press("right")
                await pilot.press("enter")
                await pilot.pause(0.1)

                assert app.multiplot is False
                assert app.active_metric() == "accuracy"

    asyncio.run(run())


def test_multiplot_selected_setting_only_changes_selected_metric() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "lightning_logs" / "version_0" / "metrics.csv"
            write_multimetric_log(metrics)

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                app.selected_metrics = ["loss", "accuracy"]
                await pilot.press("space")
                await pilot.press("right")
                await pilot.press("y")
                await pilot.pause(0.1)

                assert app.log_y is False
                assert app.metric_settings["loss"].log_y is True
                assert "accuracy" not in app.metric_settings

                await pilot.press("escape")
                await pilot.press("y")
                await pilot.pause(0.1)

                assert app.log_y is True
                assert app.metric_settings == {}

    asyncio.run(run())


def test_multiplot_right_moves_to_next_page() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "lightning_logs" / "version_0" / "metrics.csv"
            write_multimetric_log(metrics)

            app = LightningTuiApp(root)
            async with app.run_test(size=(80, 25)) as pilot:
                await pilot.pause(0.5)
                app.selected_metrics = ["loss", "accuracy", "kl"]
                await pilot.press("space")
                await pilot.press("right")
                await pilot.press("right")
                await pilot.pause(0.1)

                assert app.multiplot_page == 1
                assert app.multiplot_selection == 1

    asyncio.run(run())


def test_header_has_three_rows_without_plot_status_fields() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "lightning_logs" / "version_0" / "metrics.csv"
            write_multimetric_log(metrics)

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)

                text = str(app.query_one("#header").render())
                rows = text.splitlines()
                assert len(rows) == 3
                assert rows[0].startswith("runs:")
                assert rows[1].startswith("configs:")
                assert rows[2].startswith("metrics:")
                assert "status: active/live" not in text
                assert "metric:" not in text
                assert "mode:" not in text
                assert "status:" not in text
                assert "x-axis:" not in text

    asyncio.run(run())


def test_header_summarizes_all_runs_configs_and_metrics() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "stage3" / "version_0" / "metrics.csv"
            second = root / "stage3" / "version_1" / "metrics.csv"
            third = root / "stage4" / "mini" / "metrics.csv"
            for index, path in enumerate((first, second, third)):
                path.parent.mkdir(parents=True)
                path.write_text(f"step,train_loss,val_accuracy\n0,{1 - index * 0.1},{0.5 + index * 0.1}\n")
            (first.parent / "config.yaml").write_text("model: first\n")
            (third.parent / "config.yaml").write_text("model: third\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(140, 40)) as pilot:
                await pilot.pause(0.5)
                app.runs = [replace(run, status="active" if run.version_name == "version_1" else "stale") for run in app.runs]
                summary = app.data_summary_text()
                app.update_header()
                header = str(app.query_one("#header").render())

                assert summary.plain == "stage3 (v0, v1), stage4 (mini)"
                assert "configs: 2" in header
                assert "metrics: 2" in header
                assert any(span.style == "green" for span in summary.spans)

    asyncio.run(run())


def test_header_collapses_model_versions_after_three() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "stage3" / f"version_{index}" / "metrics.csv" for index in range(5)]
            for path in paths:
                path.parent.mkdir(parents=True)
                path.write_text("step,train_loss\n0,1.0\n")

            app = LightningTuiApp(root)
            async with app.run_test(size=(140, 40)) as pilot:
                await pilot.pause(0.5)

                assert "stage3 (v0, v1 and 3 more)" in app.data_summary_text().plain

    asyncio.run(run())


def test_app_does_not_pass_redundant_y_axis_label_to_plot() -> None:
    async def run() -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "lightning_logs" / "version_0" / "metrics.csv"
            write_multimetric_log(metrics)

            app = LightningTuiApp(root)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                with patch("ltui.app.render_plot", return_value=PlotResult("plot", ())) as render:
                    app.render_current()

                assert render.call_args.kwargs["y_label"] == ""

    asyncio.run(run())


def test_footer_starts_with_runs_configs_metrics_and_keeps_theme_near_quit() -> None:
    text = keybinding_bar().plain

    assert text.index("runs") < text.index("configs") < text.index("metrics")
    assert text.index("theme") < text.index("quit")
    assert text.index("theme") > text.index("clear")


def test_footer_distributes_available_space_and_falls_back_to_two_rows() -> None:
    wide = keybinding_bar(width=180).plain
    narrow = keybinding_bar(width=80).plain

    assert "\n" not in wide
    assert len(wide) == 180
    assert "\n" in narrow
    assert [len(row) for row in narrow.splitlines()] == [80, 80]
