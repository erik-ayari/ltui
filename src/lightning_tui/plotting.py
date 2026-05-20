from __future__ import annotations

from dataclasses import dataclass
import math

import plotext as plt


@dataclass(frozen=True)
class PlotCurve:
    label: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    color: str = "blue"
    role: str = "raw"


@dataclass(frozen=True)
class PlotResult:
    text: str
    status_messages: tuple[str, ...]


def render_plot(
    curves: list[PlotCurve],
    *,
    width: int,
    height: int,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    smoothing: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    x_min: float | None = None,
) -> PlotResult:
    prepared: list[PlotCurve] = []
    dropped_x = 0
    dropped_y = 0

    for curve in curves:
        next_curve, next_dropped_x, next_dropped_y = prepare_curve(curve, smoothing, log_x, log_y)
        dropped_x += next_dropped_x
        dropped_y += next_dropped_y
        if next_curve.x:
            prepared.append(next_curve)

    messages: list[str] = []
    if log_x and dropped_x:
        messages.append(f"log-x: dropped {dropped_x} nonpositive points")
    if log_y and dropped_y:
        messages.append(f"log-y: dropped {dropped_y} nonpositive points")

    if not prepared:
        text = "No plottable points for current selection."
        if messages:
            text += "\n" + " | ".join(messages)
        return PlotResult(text=text, status_messages=tuple(messages))

    plt.clear_figure()
    plt.plotsize(max(width, 30), max(height, 8))
    plt.grid(True, True)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(f"log10({x_label})" if log_x else x_label)
    if y_label:
        plt.ylabel(f"log10({y_label})" if log_y else y_label)
    if x_min is not None and not log_x:
        plt.xlim(left=x_min)

    for curve in prepared:
        if curve.role == "val":
            plot_dashed(curve)
        else:
            plt.plot(curve.x, curve.y, label=curve.label, color=curve.color, marker="braille")

    return PlotResult(text=plt.build(), status_messages=tuple(messages))


def prepare_curve(
    curve: PlotCurve,
    smoothing: bool,
    log_x: bool,
    log_y: bool,
) -> tuple[PlotCurve, int, int]:
    x_values = list(curve.x)
    y_values = smooth_values(list(curve.y)) if smoothing else list(curve.y)
    kept_x: list[float] = []
    kept_y: list[float] = []
    dropped_x = 0
    dropped_y = 0

    for x, y in zip(x_values, y_values, strict=False):
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        bad_x = log_x and x <= 0
        bad_y = log_y and y <= 0
        if bad_x:
            dropped_x += 1
        if bad_y:
            dropped_y += 1
        if bad_x or bad_y:
            continue
        kept_x.append(math.log10(x) if log_x else x)
        kept_y.append(math.log10(y) if log_y else y)

    return (
        PlotCurve(
            label=curve.label,
            x=tuple(kept_x),
            y=tuple(kept_y),
            color=curve.color,
            role=curve.role,
        ),
        dropped_x,
        dropped_y,
    )


def smooth_values(values: list[float], alpha: float = 0.2) -> list[float]:
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return smoothed


def plot_dashed(curve: PlotCurve) -> None:
    label = curve.label
    if len(curve.x) < 3:
        plt.plot(curve.x, curve.y, label=label, color=curve.color, marker="dot")
        return

    first = True
    for start in range(0, len(curve.x), 4):
        x_segment = curve.x[start : start + 2]
        y_segment = curve.y[start : start + 2]
        if len(x_segment) < 2:
            continue
        plt.plot(
            x_segment,
            y_segment,
            label=label if first else None,
            color=curve.color,
            marker="braille",
        )
        first = False
