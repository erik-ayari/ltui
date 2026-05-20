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
