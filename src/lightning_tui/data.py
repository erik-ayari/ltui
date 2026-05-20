from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


STEP_COLUMN = "step"
EPOCH_COLUMN = "epoch"
ROW_INDEX_AXIS = "row"
TRAIN_PREFIX = "train_"
VAL_PREFIX = "val_"


@dataclass(frozen=True)
class MetricFamily:
    name: str
    train: str | None = None
    val: str | None = None
    raw: tuple[str, ...] = ()

    def metric_names(self) -> tuple[tuple[str, str], ...]:
        metrics: list[tuple[str, str]] = []
        if self.train is not None:
            metrics.append((self.train, "train"))
        if self.val is not None:
            metrics.append((self.val, "val"))
        metrics.extend((metric, "raw") for metric in self.raw)
        return tuple(metrics)


@dataclass(frozen=True)
class RunMetrics:
    path: Path
    frame: pd.DataFrame
    x_axis: str
    metric_names: tuple[str, ...]
    families: tuple[MetricFamily, ...]


@dataclass(frozen=True)
class MetricSeries:
    metric_name: str
    x_axis: str
    x: tuple[float, ...]
    y: tuple[float, ...]


def load_run_metrics(path: str | Path) -> RunMetrics:
    metrics_path = Path(path)
    frame = read_metrics_csv(metrics_path)
    x_axis = infer_x_axis(frame)
    metrics = numeric_metric_names(frame)
    families = group_metric_families(metrics)
    return RunMetrics(
        path=metrics_path.resolve(),
        frame=frame,
        x_axis=x_axis,
        metric_names=metrics,
        families=families,
    )


def read_metrics_csv(path: str | Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError):
        return pd.DataFrame()

    frame = frame.copy()
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any() or column in {STEP_COLUMN, EPOCH_COLUMN}:
            frame[column] = converted
    return frame


def infer_x_axis(frame: pd.DataFrame) -> str:
    if has_numeric_values(frame, STEP_COLUMN):
        return STEP_COLUMN
    if has_numeric_values(frame, EPOCH_COLUMN):
        return EPOCH_COLUMN
    return ROW_INDEX_AXIS


def numeric_metric_names(frame: pd.DataFrame) -> tuple[str, ...]:
    names: list[str] = []
    for column in frame.columns:
        if column in {STEP_COLUMN, EPOCH_COLUMN}:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any():
            names.append(str(column))
    return tuple(names)


def group_metric_families(metrics: tuple[str, ...] | list[str]) -> tuple[MetricFamily, ...]:
    grouped: dict[str, dict[str, object]] = {}
    order: list[str] = []

    for metric in metrics:
        family_name, role = split_metric_family(metric)
        if family_name not in grouped:
            grouped[family_name] = {"train": None, "val": None, "raw": []}
            order.append(family_name)
        if role == "train":
            current = grouped[family_name]["train"]
            if current is None or metric_role_priority(metric, role) < metric_role_priority(current, role):
                grouped[family_name]["train"] = metric
        elif role == "val":
            current = grouped[family_name]["val"]
            if current is None or metric_role_priority(metric, role) < metric_role_priority(current, role):
                grouped[family_name]["val"] = metric
        else:
            grouped[family_name]["raw"].append(metric)

    families: list[MetricFamily] = []
    for family_name in order:
        item = grouped[family_name]
        families.append(
            MetricFamily(
                name=family_name,
                train=item["train"],
                val=item["val"],
                raw=tuple(item["raw"]),
            )
        )
    return tuple(families)


def split_metric_family(metric: str) -> tuple[str, str]:
    if metric.startswith(TRAIN_PREFIX) and len(metric) > len(TRAIN_PREFIX):
        return strip_logging_suffix(metric[len(TRAIN_PREFIX) :]), "train"
    if metric.startswith(VAL_PREFIX) and len(metric) > len(VAL_PREFIX):
        return strip_logging_suffix(metric[len(VAL_PREFIX) :]), "val"
    return metric, "raw"


def strip_logging_suffix(metric: str) -> str:
    for suffix in ("_step", "_epoch"):
        if metric.endswith(suffix) and len(metric) > len(suffix):
            return metric[: -len(suffix)]
    return metric


def metric_role_priority(metric: str, role: str) -> int:
    suffix = ""
    if metric.endswith("_step"):
        suffix = "_step"
    elif metric.endswith("_epoch"):
        suffix = "_epoch"

    if role == "train":
        return {"_step": 0, "": 1, "_epoch": 2}.get(suffix, 3)
    if role == "val":
        return {"": 0, "_epoch": 1, "_step": 2}.get(suffix, 3)
    return 0


def resolve_family(metrics: RunMetrics, family_name: str) -> tuple[tuple[str, str], ...]:
    for family in metrics.families:
        if family.name == family_name:
            return family.metric_names()
    if family_name in metrics.metric_names:
        return ((family_name, "raw"),)
    return ()


def metric_series(metrics: RunMetrics, metric_name: str) -> MetricSeries:
    frame = metrics.frame
    if metric_name not in frame.columns:
        return MetricSeries(metric_name, metrics.x_axis, (), ())

    y = pd.to_numeric(frame[metric_name], errors="coerce")
    if metrics.x_axis == ROW_INDEX_AXIS:
        x = pd.Series(range(len(frame)), dtype="float64")
    else:
        x = pd.to_numeric(frame[metrics.x_axis], errors="coerce")

    series = pd.DataFrame({"x": x, "y": y}).dropna(subset=["y"])
    if metrics.x_axis != ROW_INDEX_AXIS:
        series = series.dropna(subset=["x"])
    series = series.sort_values("x", kind="mergesort")
    return MetricSeries(
        metric_name=metric_name,
        x_axis=metrics.x_axis,
        x=tuple(float(value) for value in series["x"].tolist()),
        y=tuple(float(value) for value in series["y"].tolist()),
    )


def has_numeric_values(frame: pd.DataFrame, column: str) -> bool:
    return (
        column in frame.columns
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
    )
