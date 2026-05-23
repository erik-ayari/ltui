from __future__ import annotations

from dataclasses import dataclass
import math
import re

import plotext as plt


@dataclass(frozen=True)
class PlotCurve:
    label: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    color: str = "blue"
    role: str = "raw"
    run_label: str | None = None
    run_status: str | None = None
    style_label: str | None = None


@dataclass(frozen=True)
class PlotResult:
    text: str
    status_messages: tuple[str, ...]


@dataclass(frozen=True)
class PlotPanel:
    title: str
    curves: list[PlotCurve]
    x_label: str
    y_label: str = ""
    show_legend: bool = True
    smoothing: bool = False
    log_x: bool = False
    log_y: bool = False
    x_min: float | None = None
    dark_mode: bool = True


@dataclass(frozen=True)
class GridLayout:
    columns: int
    rows: int
    page_size: int
    page_count: int


@dataclass(frozen=True)
class GridResult:
    text: str
    status_messages: tuple[str, ...]
    layout: GridLayout
    page: int


@dataclass(frozen=True)
class PlotBounds:
    x_left: float
    x_right: float
    y_lower: float
    y_upper: float


@dataclass(frozen=True)
class LegendEntry:
    label: str
    color: str
    active: bool = False


ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")


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
    dark_mode: bool = True,
    show_legend: bool = True,
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

    bounds = plot_bounds(prepared, x_min=None if log_x else x_min)
    plt.clear_figure()
    plt.theme("dark" if dark_mode else "default")
    plt.plotsize(max(width, 30), max(height, 8))
    plt.grid(True, True)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(f"log10({x_label})" if log_x else x_label)
    if y_label:
        plt.ylabel(f"log10({y_label})" if log_y else y_label)
    plt.xlim(left=bounds.x_left, right=bounds.x_right)
    plt.ylim(lower=bounds.y_lower, upper=bounds.y_upper)
    if x_label in {"step", "epoch"} and not log_x:
        ticks = integer_ticks(bounds.x_left, bounds.x_right)
        plt.xticks(ticks, [str(tick) for tick in ticks])

    for curve in prepared:
        if curve.role == "val":
            plot_val(curve)
        else:
            plt.plot(curve.x, curve.y, label=None, color=curve.color, marker="braille")
    legend_entries = draw_legend_entries(prepared, bounds, dark_mode) if show_legend else []

    text = color_active_run_labels(plt.build(), legend_entries)
    return PlotResult(text=text, status_messages=tuple(messages))


def render_plot_grid(
    panels: list[PlotPanel],
    *,
    width: int,
    height: int,
    page: int = 0,
    selected_index: int | None = None,
) -> GridResult:
    layout = grid_layout(len(panels), width, height)
    page = min(max(page, 0), layout.page_count - 1)
    first = page * layout.page_size
    visible = panels[first : first + layout.page_size]
    gap = 1
    cell_width = max((width - gap * (layout.columns - 1)) // layout.columns, 1)
    cell_height = max((height - gap * (layout.rows - 1)) // layout.rows, 1)
    rows: list[str] = []
    messages: list[str] = []

    for row in range(layout.rows):
        cells: list[list[str]] = []
        for column in range(layout.columns):
            panel_index = first + row * layout.columns + column
            panel = visible[row * layout.columns + column] if row * layout.columns + column < len(visible) else None
            selected = selected_index == panel_index
            if panel is None:
                cells.append(empty_cell(cell_width, cell_height))
                continue
            rendered = render_plot(
                panel.curves,
                width=max(cell_width - 2, 30),
                height=max(cell_height - 2, 8),
                title=panel.title,
                x_label=panel.x_label,
                y_label=panel.y_label,
                smoothing=panel.smoothing,
                log_x=panel.log_x,
                log_y=panel.log_y,
                x_min=panel.x_min,
                dark_mode=panel.dark_mode,
                show_legend=row == 0 and column == 0,
            )
            messages.extend(rendered.status_messages)
            cells.append(frame_cell(rendered.text, cell_width, cell_height, selected))
        for line_index in range(cell_height):
            rows.append((" " * gap).join(cell[line_index] for cell in cells))
        if row < layout.rows - 1:
            rows.extend("" for _ in range(gap))

    return GridResult(
        text="\n".join(rows),
        status_messages=tuple(dict.fromkeys(messages)),
        layout=layout,
        page=page,
    )


def grid_layout(total: int, width: int, height: int) -> GridLayout:
    total = max(total, 1)
    min_width = 44
    min_height = 14
    max_columns = max(1, width // min_width)
    max_rows = max(1, height // min_height)
    capacity = max(1, max_columns * max_rows)
    page_size = min(total, capacity)
    best: tuple[float, int, int] | None = None

    for columns in range(1, max_columns + 1):
        rows = math.ceil(page_size / columns)
        if rows > max_rows:
            continue
        cell_width = width / columns
        cell_height = height / rows
        shape_score = abs((cell_width / max(cell_height, 1)) - 3.2)
        empty_score = (columns * rows - page_size) * 0.35
        score = shape_score + empty_score
        if best is None or score < best[0]:
            best = (score, columns, rows)

    if best is None:
        columns = max_columns
        rows = max_rows
    else:
        columns = best[1]
        rows = best[2]
    return GridLayout(
        columns=columns,
        rows=rows,
        page_size=columns * rows,
        page_count=math.ceil(total / (columns * rows)),
    )


def frame_cell(text: str, width: int, height: int, selected: bool) -> list[str]:
    inner_width = max(width - 2, 1)
    inner_height = max(height - 2, 1)
    lines = fit_lines(text.splitlines(), inner_width, inner_height)
    if not selected:
        return [fit_plain("", width)] + [f" {line} " for line in lines] + [fit_plain("", width)]

    accent = "\033[38;5;214m"
    reset = "\033[39m"
    top = f"{accent}┌{'─' * inner_width}┐{reset}"
    bottom = f"{accent}└{'─' * inner_width}┘{reset}"
    return [top] + [f"{accent}│{reset}{line}{accent}│{reset}" for line in lines] + [bottom]


def empty_cell(width: int, height: int) -> list[str]:
    return [fit_plain("", width) for _ in range(height)]


def fit_lines(lines: list[str], width: int, height: int) -> list[str]:
    fitted = [fit_ansi_line(line, width) for line in lines[:height]]
    while len(fitted) < height:
        fitted.append(fit_plain("", width))
    return fitted


def fit_ansi_line(line: str, width: int) -> str:
    visible = visible_length(line)
    if visible > width:
        return fit_plain(strip_ansi(line), width)
    return line + " " * (width - visible)


def fit_plain(line: str, width: int) -> str:
    return line[:width].ljust(width)


def visible_length(line: str) -> int:
    return len(strip_ansi(line))


def strip_ansi(line: str) -> str:
    return ansi_pattern.sub("", line)


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
            run_label=curve.run_label,
            run_status=curve.run_status,
            style_label=curve.style_label,
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


def plot_bounds(curves: list[PlotCurve], x_min: float | None = None) -> PlotBounds:
    x_values = [value for curve in curves for value in curve.x]
    y_values = [value for curve in curves for value in curve.y]
    x_left = min(x_values)
    x_right = max(x_values)
    if x_min is not None:
        x_left = x_min
    else:
        x_pad = axis_padding(x_left, x_right)
        x_left -= x_pad
        x_right += x_pad

    if x_right <= x_left:
        x_right = x_left + 1
    elif x_min is not None:
        x_right += axis_padding(x_left, x_right)

    y_lower = min(y_values)
    y_upper = max(y_values)
    y_pad = axis_padding(y_lower, y_upper)
    return PlotBounds(
        x_left=x_left,
        x_right=x_right,
        y_lower=y_lower - y_pad,
        y_upper=y_upper + y_pad,
    )


def axis_padding(lower: float, upper: float) -> float:
    span = upper - lower
    if span > 0:
        return span * 0.03
    magnitude = max(abs(lower), abs(upper), 1.0)
    return magnitude * 0.5


def integer_ticks(left: float, right: float, target_count: int = 6) -> list[int]:
    start = math.ceil(left)
    end = math.floor(right)
    if end < start:
        return [round(left)]

    span = max(end - start, 1)
    step = nice_integer_step(math.ceil(span / max(target_count - 1, 1)))
    first = math.ceil(start / step) * step
    ticks = list(range(first, end + 1, step))
    if start == 0 and (not ticks or ticks[0] != 0):
        ticks.insert(0, 0)
    return ticks or [start]


def nice_integer_step(minimum: int) -> int:
    if minimum <= 1:
        return 1
    magnitude = 10 ** int(math.floor(math.log10(minimum)))
    for factor in (1, 2, 5, 10):
        step = factor * magnitude
        if step >= minimum:
            return step
    return 10 * magnitude


def plot_val(curve: PlotCurve) -> None:
    plt.plot(curve.x, curve.y, label=None, color=curve.color, marker="dot")


def draw_legend_entries(curves: list[PlotCurve], bounds: PlotBounds, dark_mode: bool) -> list[LegendEntry]:
    x = bounds.x_left
    y = bounds.y_upper
    entries = run_legend_entries(curves)
    neutral = "white" if dark_mode else "black"

    for entry in entries:
        plt.plot([x], [y], label=entry.label, color=entry.color, marker="dot")
    if any(curve.style_label == "train" for curve in curves):
        plt.plot([x], [y], label="train", color=neutral, marker="braille")
    if any(curve.style_label == "val" for curve in curves):
        plt.plot([x], [y], label="val", color=neutral, marker="dot")
    return entries


def run_legend_entries(curves: list[PlotCurve]) -> list[LegendEntry]:
    entries: list[LegendEntry] = []
    seen: set[str] = set()
    for curve in curves:
        if curve.run_label is None or curve.run_label in seen:
            continue
        entries.append(LegendEntry(curve.run_label, curve.color, curve.run_status == "active"))
        seen.add(curve.run_label)
    return entries


def color_active_run_labels(text: str, entries: list[LegendEntry]) -> str:
    for entry in entries:
        if entry.active:
            text = text.replace(entry.label, f"\033[38;5;10m{entry.label}\033[39m", 1)
    return text
