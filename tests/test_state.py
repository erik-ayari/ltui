from pathlib import Path

from lightning_tui.discovery import RunVersion
from lightning_tui.state import UiState, valid_state


def test_state_restore_drops_missing_paths_and_metrics(tmp_path: Path) -> None:
    existing = tmp_path / "version_0" / "metrics.csv"
    existing.parent.mkdir(parents=True)
    existing.write_text("step,loss\n0,1.0\n")
    run = RunVersion(
        display_name="version_0",
        metrics_csv_path=existing.resolve(),
        parent_name=None,
        version_name="version_0",
        last_modified=1.0,
        status="active",
        available_numeric_metrics=("loss",),
    )
    saved = UiState(
        selected_run_paths=(str(existing.resolve()), str(tmp_path / "missing.csv")),
        selected_metrics=("loss", "missing"),
        active_metric="missing",
        x_axis_mode="epoch",
    )

    restored = valid_state(saved, [run], ("loss",))

    assert restored.selected_run_paths == (str(existing.resolve()),)
    assert restored.selected_metrics == ("loss",)
    assert restored.active_metric == "loss"
    assert restored.x_axis_mode == "epoch"
