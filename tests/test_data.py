from pathlib import Path

import pandas as pd

from lightning_tui.data import (
    ROW_INDEX_AXIS,
    group_metric_families,
    infer_x_axis,
    load_run_metrics,
    metric_series,
    resolve_family,
)


def test_extracts_numeric_metrics_excluding_step_epoch(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("step,epoch,train_loss,val_loss,label\n0,0,1.0,,a\n1,0,,0.8,b\n")

    metrics = load_run_metrics(path)

    assert metrics.metric_names == ("train_loss", "val_loss")


def test_drops_nan_metric_rows_and_sorts_by_x(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("step,train_loss\n3,0.3\n1,\n2,0.2\n")

    metrics = load_run_metrics(path)
    series = metric_series(metrics, "train_loss")

    assert series.x == (2.0, 3.0)
    assert series.y == (0.2, 0.3)


def test_x_axis_fallback_order() -> None:
    assert infer_x_axis(pd.DataFrame({"step": [1], "epoch": [2], "loss": [3]})) == "step"
    assert infer_x_axis(pd.DataFrame({"epoch": [2], "loss": [3]})) == "epoch"
    assert infer_x_axis(pd.DataFrame({"loss": [3]})) == ROW_INDEX_AXIS


def test_groups_train_val_metrics_into_families() -> None:
    families = group_metric_families(("train_loss", "val_loss", "train_recon_loss", "val_kl"))

    by_name = {family.name: family for family in families}
    assert by_name["loss"].train == "train_loss"
    assert by_name["loss"].val == "val_loss"
    assert by_name["recon_loss"].train == "train_recon_loss"
    assert by_name["kl"].val == "val_kl"


def test_resolve_family_handles_missing_train_or_val(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("step,val_loss\n0,1.0\n")

    metrics = load_run_metrics(path)

    assert resolve_family(metrics, "loss") == (("val_loss", "val"),)
