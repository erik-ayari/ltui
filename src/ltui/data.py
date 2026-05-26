from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import pandas as pd


STEP_COLUMN = "step"
EPOCH_COLUMN = "epoch"
ROW_INDEX_AXIS = "row"
TRAIN_PREFIX = "train_"
VAL_PREFIX = "val_"
LTUI_MANIFEST = "ltui_manifest.json"
AxisMode = Literal["step", "epoch", "row"]


@dataclass(frozen=True)
class MetricFamily:
    name: str
    train: str | None = None
    val: str | None = None
    raw: tuple[str, ...] = ()
    train_variants: tuple[str, ...] = ()
    val_variants: tuple[str, ...] = ()

    def metric_names(self, x_axis: AxisMode = STEP_COLUMN) -> tuple[tuple[str, str], ...]:
        metrics: list[tuple[str, str]] = []
        train = choose_metric_variant(self.train_variants, "train", x_axis)
        val = choose_metric_variant(self.val_variants, "val", x_axis)
        if train is not None:
            metrics.append((train, "train"))
        if val is not None:
            metrics.append((val, "val"))
        metrics.extend((metric, "raw") for metric in self.raw)
        return tuple(metrics)


@dataclass(frozen=True)
class StructuredMetricSource:
    name: str
    role: str
    metric_path: tuple[str, ...]
    csv_path: Path


@dataclass(frozen=True)
class ImageSource:
    name: str
    role: str
    image_path: tuple[str, ...]
    directory: Path


@dataclass(frozen=True)
class RunMetrics:
    path: Path
    frame: pd.DataFrame
    x_axis: str
    metric_names: tuple[str, ...]
    families: tuple[MetricFamily, ...]
    structured_sources: tuple[StructuredMetricSource, ...] = ()
    image_sources: tuple[ImageSource, ...] = ()


@dataclass(frozen=True)
class MetricSeries:
    metric_name: str
    x_axis: str
    x: tuple[float, ...]
    y: tuple[float, ...]


def load_run_metrics(path: str | Path) -> RunMetrics:
    metrics_path = Path(path)
    if metrics_path.name == LTUI_MANIFEST:
        return load_structured_run_metrics(metrics_path)

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


def load_structured_run_metrics(path: Path) -> RunMetrics:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        manifest = {}

    sources: list[StructuredMetricSource] = []
    for item in manifest.get("series", ()):
        if not isinstance(item, dict):
            continue
        metric_path = item.get("metric_path")
        source_path = item.get("path")
        name = item.get("name")
        role = item.get("role")
        if not isinstance(metric_path, list) or not isinstance(source_path, str) or not isinstance(name, str) or not isinstance(role, str):
            continue
        parts = tuple(str(part) for part in metric_path if str(part))
        if not parts:
            continue
        sources.append(
            StructuredMetricSource(
                name=name,
                role=role if role in {"train", "val"} else "raw",
                metric_path=parts,
                csv_path=(path.parent / source_path).resolve(),
            )
        )

    image_sources = load_image_sources(manifest, path)
    metric_names = tuple(dict.fromkeys("/".join(source.metric_path) for source in sources))
    families = group_structured_metric_families(sources)
    return RunMetrics(
        path=path.resolve(),
        frame=pd.DataFrame(),
        x_axis=STEP_COLUMN,
        metric_names=metric_names,
        families=families,
        structured_sources=tuple(sources),
        image_sources=tuple(image_sources),
    )


def load_image_sources(manifest: dict, path: Path) -> list[ImageSource]:
    sources: list[ImageSource] = []
    for item in manifest.get("images", ()):
        if not isinstance(item, dict):
            continue
        image_path = item.get("image_path")
        source_path = item.get("path")
        name = item.get("name")
        role = item.get("role")
        if not isinstance(image_path, list) or not isinstance(source_path, str) or not isinstance(name, str) or not isinstance(role, str):
            continue
        parts = tuple(str(part) for part in image_path if str(part))
        if not parts:
            continue
        sources.append(
            ImageSource(
                name=name,
                role=role if role in {"train", "val"} else "raw",
                image_path=parts,
                directory=(path.parent / source_path).resolve(),
            )
        )
    return sources


def group_structured_metric_families(sources: list[StructuredMetricSource]) -> tuple[MetricFamily, ...]:
    grouped: dict[str, dict[str, list[str]]] = {}
    order: list[str] = []
    for source in sources:
        family_name = "/".join(source.metric_path)
        if family_name not in grouped:
            grouped[family_name] = {"train": [], "val": [], "raw": []}
            order.append(family_name)
        grouped[family_name][source.role].append(source.name)

    families: list[MetricFamily] = []
    for family_name in order:
        item = grouped[family_name]
        train_variants = tuple(item["train"])
        val_variants = tuple(item["val"])
        families.append(
            MetricFamily(
                name=family_name,
                train=train_variants[0] if train_variants else None,
                val=val_variants[0] if val_variants else None,
                raw=tuple(item["raw"]),
                train_variants=train_variants,
                val_variants=val_variants,
            )
        )
    return tuple(families)


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
    grouped: dict[str, dict[str, list[str]]] = {}
    order: list[str] = []

    for metric in metrics:
        family_name, role = split_metric_family(metric)
        if family_name not in grouped:
            grouped[family_name] = {"train": [], "val": [], "raw": []}
            order.append(family_name)
        if role == "train":
            grouped[family_name]["train"].append(metric)
        elif role == "val":
            grouped[family_name]["val"].append(metric)
        else:
            grouped[family_name]["raw"].append(metric)

    families: list[MetricFamily] = []
    for family_name in order:
        item = grouped[family_name]
        train_variants = tuple(item["train"])
        val_variants = tuple(item["val"])
        families.append(
            MetricFamily(
                name=family_name,
                train=choose_metric_variant(train_variants, "train", STEP_COLUMN),
                val=choose_metric_variant(val_variants, "val", STEP_COLUMN),
                raw=tuple(item["raw"]),
                train_variants=train_variants,
                val_variants=val_variants,
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


def choose_metric_variant(metrics: tuple[str, ...], role: str, x_axis: AxisMode) -> str | None:
    if not metrics:
        return None
    return min(metrics, key=lambda metric: metric_role_priority(metric, role, x_axis))


def metric_role_priority(metric: str, role: str, x_axis: AxisMode) -> int:
    suffix = ""
    if metric.endswith("_step"):
        suffix = "_step"
    elif metric.endswith("_epoch"):
        suffix = "_epoch"

    if x_axis == EPOCH_COLUMN:
        return {"_epoch": 0, "": 1, "_step": 2}.get(suffix, 3)
    if role == "train":
        return {"_step": 0, "": 1, "_epoch": 2}.get(suffix, 3)
    if role == "val":
        return {"": 0, "_epoch": 1, "_step": 2}.get(suffix, 3)
    return 0


def resolve_family(
    metrics: RunMetrics,
    family_name: str,
    x_axis: AxisMode = STEP_COLUMN,
) -> tuple[tuple[str, str], ...]:
    for family in metrics.families:
        if family.name == family_name:
            return family.metric_names(x_axis)
    if family_name in metrics.metric_names:
        return ((family_name, "raw"),)
    return ()


def metric_series(metrics: RunMetrics, metric_name: str, x_axis: AxisMode | None = None) -> MetricSeries:
    if metrics.structured_sources:
        return structured_metric_series(metrics, metric_name, x_axis)

    frame = metrics.frame
    if metric_name not in frame.columns:
        return MetricSeries(metric_name, metrics.x_axis, (), ())

    resolved_x_axis = resolve_x_axis(frame, x_axis or metrics.x_axis)
    y = pd.to_numeric(frame[metric_name], errors="coerce")
    if resolved_x_axis == ROW_INDEX_AXIS:
        x = pd.Series(range(len(frame)), dtype="float64")
    else:
        x = pd.to_numeric(frame[resolved_x_axis], errors="coerce")

    series = pd.DataFrame({"x": x, "y": y}).dropna(subset=["y"])
    if resolved_x_axis != ROW_INDEX_AXIS:
        series = series.dropna(subset=["x"])
    series = series.sort_values("x", kind="mergesort")
    return MetricSeries(
        metric_name=metric_name,
        x_axis=resolved_x_axis,
        x=tuple(float(value) for value in series["x"].tolist()),
        y=tuple(float(value) for value in series["y"].tolist()),
    )


def structured_metric_series(metrics: RunMetrics, metric_name: str, x_axis: AxisMode | None = None) -> MetricSeries:
    source = next((item for item in metrics.structured_sources if item.name == metric_name), None)
    if source is None:
        return MetricSeries(metric_name, metrics.x_axis, (), ())

    frame = read_metrics_csv(source.csv_path)
    resolved_x_axis = resolve_x_axis(frame, x_axis or metrics.x_axis)
    if "value" not in frame.columns:
        return MetricSeries(metric_name, resolved_x_axis, (), ())

    y = pd.to_numeric(frame["value"], errors="coerce")
    if resolved_x_axis == ROW_INDEX_AXIS:
        x = pd.Series(range(len(frame)), dtype="float64")
    else:
        x = pd.to_numeric(frame[resolved_x_axis], errors="coerce")

    series = pd.DataFrame({"x": x, "y": y}).dropna(subset=["y"])
    if resolved_x_axis != ROW_INDEX_AXIS:
        series = series.dropna(subset=["x"])
    series = series.sort_values("x", kind="mergesort")
    return MetricSeries(
        metric_name=metric_name,
        x_axis=resolved_x_axis,
        x=tuple(float(value) for value in series["x"].tolist()),
        y=tuple(float(value) for value in series["y"].tolist()),
    )


def resolve_x_axis(frame: pd.DataFrame, preferred: AxisMode) -> AxisMode:
    if preferred != ROW_INDEX_AXIS and has_numeric_values(frame, preferred):
        return preferred
    if has_numeric_values(frame, STEP_COLUMN):
        return STEP_COLUMN
    if has_numeric_values(frame, EPOCH_COLUMN):
        return EPOCH_COLUMN
    return ROW_INDEX_AXIS


def has_numeric_values(frame: pd.DataFrame, column: str) -> bool:
    return (
        column in frame.columns
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
    )
