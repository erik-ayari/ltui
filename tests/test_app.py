import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from lightning_tui.app import LightningTuiApp, SelectorScreen


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

                assert [curve.label for curve in app.build_curves()] == ["run_a/version_0 train", "run_b/version_0 train"]

    asyncio.run(run())
