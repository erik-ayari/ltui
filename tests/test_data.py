from pathlib import Path

import pandas as pd

from lightning_tui.data import (
    EPOCH_COLUMN,
    ROW_INDEX_AXIS,
    STEP_COLUMN,
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


def test_groups_lightning_step_epoch_suffixes_into_base_family() -> None:
    families = group_metric_families(("train_loss_epoch", "train_loss_step", "val_loss"))

    by_name = {family.name: family for family in families}
    assert by_name["loss"].train == "train_loss_step"
    assert by_name["loss"].val == "val_loss"
    assert set(by_name["loss"].train_variants) == {"train_loss_epoch", "train_loss_step"}
    assert "loss_step" not in by_name
    assert "loss_epoch" not in by_name


def test_resolve_family_switches_train_step_epoch_variant(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,step,train_loss_step,train_loss_epoch,val_loss\n0,10,0.9,,\n0,20,,0.8,\n0,20,,,0.7\n")

    metrics = load_run_metrics(path)

    assert resolve_family(metrics, "loss", STEP_COLUMN) == (("train_loss_step", "train"), ("val_loss", "val"))
    assert resolve_family(metrics, "loss", EPOCH_COLUMN) == (("train_loss_epoch", "train"), ("val_loss", "val"))


def test_metric_series_uses_requested_x_axis_for_validation_alignment(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,step,train_loss_step,val_loss\n0,10,0.9,\n0,20,0.8,\n0,20,,0.7\n")

    metrics = load_run_metrics(path)
    step_series = metric_series(metrics, "val_loss", STEP_COLUMN)
    epoch_series = metric_series(metrics, "val_loss", EPOCH_COLUMN)

    assert step_series.x == (20.0,)
    assert epoch_series.x == (0.0,)


def test_resolve_family_handles_missing_train_or_val(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("step,val_loss\n0,1.0\n")

    metrics = load_run_metrics(path)

    assert resolve_family(metrics, "loss") == (("val_loss", "val"),)
