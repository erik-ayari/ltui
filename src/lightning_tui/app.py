from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from rapidfuzz import fuzz
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from .data import AxisMode, RunMetrics, load_run_metrics, metric_series, resolve_family, resolve_x_axis
from .discovery import RunVersion, discover_runs
from .plotting import PlotCurve, render_plot
from .state import UiState, load_state, save_state, valid_state


PALETTE = ("blue+", "orange+", "cyan+", "magenta+", "green+", "red+", "white", "gray+")


class SelectorItem:
    def __init__(self, key: str, label: str, search_text: str | None = None) -> None:
        self.key = key
        self.label = label
        self.search_text = search_text or label


class SelectorScreen(ModalScreen[list[str] | None]):
    BINDINGS = [
        Binding("/", "focus_search", "Search", show=False),
        Binding("space", "toggle", "Toggle", show=False),
        Binding("enter", "apply", "Apply", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, title: str, items: list[SelectorItem], selected: list[str]) -> None:
        super().__init__()
        self.title = title
        self.items = items
        self.selected = list(selected)
        self.search_query = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="selector"):
            yield Static(self.title, id="selector-title")
            yield Input(placeholder="Search", id="search")
            yield OptionList(id="options", markup=False)
            yield Static("/ search  arrows navigate  space toggle  enter apply  escape cancel", id="selector-help")

    def on_mount(self) -> None:
        self.refresh_options()
        self.query_one(OptionList).focus()

    def action_focus_search(self) -> None:
        self.query_one(Input).focus()

    def action_toggle(self) -> None:
        option_list = self.query_one(OptionList)
        if option_list.highlighted is None:
            return
        option = option_list.get_option_at_index(option_list.highlighted)
        if option.id is None:
            return
        key = str(option.id)
        if key in self.selected:
            self.selected.remove(key)
        else:
            self.selected.append(key)
        self.refresh_options(highlight_key=key)

    def action_apply(self) -> None:
        option_list = self.query_one(OptionList)
        if not self.selected and option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            if option.id is not None:
                self.selected.append(str(option.id))
        self.dismiss(self.selected)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.action_apply()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search":
            return
        self.search_query = event.value
        self.refresh_options()

    def refresh_options(self, highlight_key: str | None = None) -> None:
        option_list = self.query_one(OptionList)
        visible_items = self.filtered_items()
        option_list.clear_options()
        options = [Option(self.format_label(item), id=item.key) for item in visible_items]
        option_list.add_options(options)
        if not options:
            return
        highlight = 0
        if highlight_key is not None:
            for index, item in enumerate(visible_items):
                if item.key == highlight_key:
                    highlight = index
                    break
        option_list.highlighted = highlight

    def filtered_items(self) -> list[SelectorItem]:
        if not self.search_query:
            return self.items
        scored = [
            (fuzz.WRatio(self.search_query, item.search_text), index, item)
            for index, item in enumerate(self.items)
        ]
        return [item for score, index, item in sorted(scored, key=lambda value: (-value[0], value[1])) if score >= 35]

    def format_label(self, item: SelectorItem) -> str:
        marker = "[x]" if item.key in self.selected else "[ ]"
        return f"{marker} {item.label}"


class ConfigItem:
    def __init__(self, run: RunVersion) -> None:
        self.run = run
        self.key = str(run.metrics_csv_path)
        self.label = run.display_name
        self.search_text = f"{run.display_name} {run.metrics_csv_path} {run.config_yaml_path}"


class ConfigScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("/", "focus_search", "Search", show=False),
        Binding("enter", "close", "Close", show=False),
        Binding("escape", "close", "Close", show=False),
    ]

    def __init__(self, items: list[ConfigItem]) -> None:
        super().__init__()
        self.items = items
        self.search_query = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="config-viewer"):
            yield Static("Configs", id="config-title")
            with Horizontal(id="config-body"):
                with Vertical(id="config-list-pane"):
                    yield Input(placeholder="Search", id="config-search")
                    yield OptionList(id="config-options", markup=False)
                yield TextArea(
                    "",
                    language="yaml",
                    read_only=True,
                    show_line_numbers=True,
                    id="config-preview",
                )
            yield Static("/ search  arrows choose run  enter/escape close", id="config-help")

    def on_mount(self) -> None:
        self.refresh_options()
        self.query_one(OptionList).focus()
        self.update_preview()

    def action_focus_search(self) -> None:
        self.query_one(Input).focus()

    def action_close(self) -> None:
        self.dismiss(None)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "config-search":
            return
        self.search_query = event.value
        self.refresh_options()
        self.update_preview()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        self.update_preview()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.update_preview()

    def refresh_options(self) -> None:
        option_list = self.query_one(OptionList)
        option_list.clear_options()
        option_list.add_options(Option(item.label, id=item.key) for item in self.filtered_items())
        if option_list.option_count:
            option_list.highlighted = 0

    def filtered_items(self) -> list[ConfigItem]:
        if not self.search_query:
            return self.items
        scored = [
            (fuzz.WRatio(self.search_query, item.search_text), index, item)
            for index, item in enumerate(self.items)
        ]
        return [item for score, index, item in sorted(scored, key=lambda value: (-value[0], value[1])) if score >= 35]

    def update_preview(self) -> None:
        text_area = self.query_one(TextArea)
        selected = self.selected_item()
        if selected is None or selected.run.config_yaml_path is None:
            text_area.load_text("No config selected.")
            return
        try:
            text = selected.run.config_yaml_path.read_text(errors="replace")
        except OSError as exc:
            text = f"Could not read {selected.run.config_yaml_path}\n\n{exc}"
        text_area.load_text(f"# {selected.run.display_name}\n# {selected.run.config_yaml_path}\n\n{text}")

    def selected_item(self) -> ConfigItem | None:
        option_list = self.query_one(OptionList)
        if option_list.highlighted is None:
            return None
        option = option_list.get_option_at_index(option_list.highlighted)
        if option.id is None:
            return None
        key = str(option.id)
        return next((item for item in self.items if item.key == key), None)


class LightningTuiApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #header {
        height: 3;
        padding: 0 1;
        background: $surface;
    }

    #plot {
        height: 1fr;
        padding: 0 1;
        background: $surface;
    }

    #footer {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }

    SelectorScreen {
        align: center middle;
    }

    ConfigScreen {
        align: center middle;
    }

    #selector {
        width: 82%;
        height: 82%;
        border: solid $accent;
        background: $panel;
    }

    #selector-title {
        height: 1;
        padding: 0 1;
    }

    #search {
        margin: 0 1;
    }

    #options {
        height: 1fr;
        margin: 0 1;
    }

    #selector-help {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    #config-viewer {
        width: 88%;
        height: 86%;
        border: solid $accent;
        background: $panel;
    }

    #config-title {
        height: 1;
        padding: 0 1;
    }

    #config-body {
        height: 1fr;
    }

    #config-list-pane {
        width: 32%;
        height: 1fr;
        padding: 0 1;
    }

    #config-search {
        height: 3;
    }

    #config-options {
        height: 1fr;
    }

    #config-preview {
        width: 1fr;
        height: 1fr;
        border-left: solid $primary;
    }

    #config-help {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("m", "open_metric_selector", "Metrics"),
        Binding("r", "open_run_selector", "Runs"),
        Binding("c", "open_config_selector", "Configs"),
        Binding("n", "next_metric", "Next metric"),
        Binding("p", "previous_metric", "Previous metric"),
        Binding("a", "toggle_x_axis", "X axis"),
        Binding("d", "toggle_dark_mode", "Dark"),
        Binding("s", "toggle_smoothing", "Smoothing"),
        Binding("x", "toggle_log_x", "Log x"),
        Binding("y", "toggle_log_y", "Log y"),
        Binding("space", "toggle_pause", "Pause"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.runs: list[RunVersion] = []
        self.metrics_cache: dict[str, RunMetrics] = {}
        self.selected_run_paths: list[str] = []
        self.selected_metrics: list[str] = []
        self.active_metric_index = 0
        self.grouped_mode = True
        self.x_axis_mode: AxisMode = "step"
        self.dark_mode = True
        self.smoothing = False
        self.log_x = False
        self.log_y = False
        self.paused = False
        self.status_message = "starting"
        self.refreshing = False

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        yield Static("", id="plot")
        yield Static("", id="footer")

    def on_mount(self) -> None:
        self.query_one("#footer", Static).update(keybinding_bar())
        asyncio.create_task(self.refresh_snapshot(initial=True))
        self.set_interval(1.5, self.schedule_live_refresh)

    def on_key(self, event) -> None:
        if event.character == "R":
            event.stop()
            asyncio.create_task(self.refresh_snapshot(force=True))

    def schedule_live_refresh(self) -> None:
        if not self.paused and not self.refreshing:
            asyncio.create_task(self.refresh_snapshot())

    async def refresh_snapshot(self, *, initial: bool = False, force: bool = False) -> None:
        if self.refreshing:
            return
        self.refreshing = True
        if force:
            self.status_message = "rescanning"
            self.render_current()
        try:
            runs, metrics_cache = await asyncio.to_thread(build_snapshot, self.root)
            self.apply_snapshot(runs, metrics_cache, initial)
            self.status_message = "live" if not self.paused else "paused"
        except Exception as exc:
            self.status_message = f"error: {exc}"
        finally:
            self.refreshing = False
            self.render_current()

    def apply_snapshot(
        self,
        runs: list[RunVersion],
        metrics_cache: dict[str, RunMetrics],
        initial: bool,
    ) -> None:
        self.runs = runs
        self.metrics_cache = metrics_cache
        if not runs:
            self.selected_run_paths = []
            self.selected_metrics = []
            self.active_metric_index = 0
            return

        if initial:
            saved = load_state(self.root)
            if saved is not None:
                self.grouped_mode = saved.grouped_mode
                self.x_axis_mode = saved.x_axis_mode
                self.dark_mode = saved.dark_mode
                self.smoothing = saved.smoothing
                self.log_x = saved.log_x
                self.log_y = saved.log_y
                choices = self.metric_choices_for_runs(
                    [str(run.metrics_csv_path) for run in runs if str(run.metrics_csv_path) in saved.selected_run_paths]
                )
                restored = valid_state(saved, runs, choices)
                self.selected_run_paths = list(restored.selected_run_paths)
                self.selected_metrics = list(restored.selected_metrics)
                if restored.active_metric in self.selected_metrics:
                    self.active_metric_index = self.selected_metrics.index(restored.active_metric)

        self.selected_run_paths = [path for path in self.selected_run_paths if path in self.metrics_cache]
        if not self.selected_run_paths:
            latest = max(runs, key=lambda run: run.last_modified)
            self.selected_run_paths = [str(latest.metrics_csv_path)]

        choices = self.metric_choices_for_runs(self.selected_run_paths)
        self.selected_metrics = [metric for metric in self.selected_metrics if metric in choices]
        if not self.selected_metrics and choices:
            self.selected_metrics = [self.default_metric_choice(choices)]
        if self.active_metric_index >= len(self.selected_metrics):
            self.active_metric_index = 0

    def render_current(self) -> None:
        plot = self.query_one("#plot", Static)
        if not self.runs:
            self.update_header()
            plot.update(self.empty_text())
            return
        if not self.selected_metrics:
            self.update_header()
            plot.update("No numeric metrics found in selected run.")
            return

        curves = self.build_curves()
        size = plot.size
        active_metric = self.active_metric()
        x_axis = self.current_x_axis()
        result = render_plot(
            curves,
            width=max(size.width - 2, 60),
            height=max(size.height - 2, 16),
            title=active_metric or "",
            x_label=x_axis,
            y_label=active_metric,
            smoothing=self.smoothing,
            log_x=self.log_x,
            log_y=self.log_y,
            x_min=0 if x_axis in {"step", "epoch"} else None,
            dark_mode=self.dark_mode,
        )
        if result.status_messages:
            self.status_message = " | ".join(result.status_messages)
        elif self.status_message.startswith("log-"):
            self.status_message = "paused" if self.paused else "live"
        self.update_header()
        plot.update(Text.from_ansi(result.text))
        self.persist_state()

    def update_header(self) -> None:
        header = self.query_one("#header", Static)
        run_names = [self.run_by_path(path).display_name for path in self.selected_run_paths if self.run_by_path(path)]
        run_text = ", ".join(run_names[:3]) if run_names else "none"
        if len(run_names) > 3:
            run_text += f" +{len(run_names) - 3}"
        statuses = sorted({self.run_by_path(path).status for path in self.selected_run_paths if self.run_by_path(path)})
        status_text = "/".join(statuses) if statuses else "no runs"
        live = "paused" if self.paused else "live"
        run_mode = "compare" if len(self.selected_run_paths) > 1 else "single"
        header.update(
            f"runs: {run_text}\n"
            f"metric: {self.active_metric() or 'none'}  mode: {run_mode}  x-axis: {self.current_x_axis()}  theme: {theme_name(self.dark_mode)}  status: {status_text}/{live}  smooth: {onoff(self.smoothing)}  log-x: {onoff(self.log_x)}  log-y: {onoff(self.log_y)}\n"
            f"{self.status_message}"
        )

    def empty_text(self) -> str:
        return (
            f"No metrics.csv files found under {self.root}\n\n"
            "Expected examples:\n"
            "  lightning_logs/version_0/metrics.csv\n"
            "  run_a/version_0/metrics.csv\n"
            "  run_a/lightning_logs/version_0/metrics.csv\n\n"
            "Press R to rescan or q to quit."
        )

    def build_curves(self) -> list[PlotCurve]:
        curves: list[PlotCurve] = []
        active_metric = self.active_metric()
        if active_metric is None:
            return curves

        for run_index, run_path in enumerate(self.selected_run_paths):
            metrics = self.metrics_cache.get(run_path)
            run = self.run_by_path(run_path)
            if metrics is None or run is None:
                continue
            color = PALETTE[run_index % len(PALETTE)]
            metric_roles = (
                resolve_family(metrics, active_metric, self.x_axis_mode)
                if self.grouped_mode
                else ((active_metric, "raw"),)
            )
            for metric_name, role in metric_roles:
                series = metric_series(metrics, metric_name, self.x_axis_mode)
                if not series.x:
                    continue
                label = self.curve_label(run, metric_name, role)
                curves.append(
                    PlotCurve(
                        label=label,
                        x=series.x,
                        y=series.y,
                        color=color,
                        role=role,
                        run_label=run.display_name,
                        run_status=run.status,
                        style_label=self.style_label(metric_name, role),
                    )
                )
        return curves

    def curve_label(self, run: RunVersion, metric_name: str, role: str) -> str:
        style_label = self.style_label(metric_name, role)
        if self.grouped_mode and role in {"train", "val"}:
            return style_label if len(self.selected_run_paths) == 1 else f"{run.display_name} {style_label}"
        return metric_name if len(self.selected_run_paths) == 1 else f"{run.display_name} {metric_name}"

    def style_label(self, metric_name: str, role: str) -> str:
        if self.grouped_mode and role in {"train", "val"}:
            return role
        return metric_name

    def metric_choices_for_runs(self, run_paths: list[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        choices: list[str] = []
        for run_path in run_paths:
            metrics = self.metrics_cache.get(run_path)
            if metrics is None:
                continue
            names = (family.name for family in metrics.families) if self.grouped_mode else iter(metrics.metric_names)
            for name in names:
                if name not in seen:
                    seen.add(name)
                    choices.append(name)
        return tuple(choices)

    def default_metric_choice(self, choices: tuple[str, ...]) -> str:
        raw_metrics = self.raw_metrics_for_selected_runs()
        if self.grouped_mode:
            if "loss" in choices and ("train_loss" in raw_metrics or "val_loss" in raw_metrics):
                return "loss"
            if "loss" in choices:
                return "loss"
        else:
            for metric in ("val_loss", "train_loss", "loss"):
                if metric in choices:
                    return metric
        return choices[0]

    def raw_metrics_for_selected_runs(self) -> set[str]:
        raw: set[str] = set()
        for run_path in self.selected_run_paths:
            metrics = self.metrics_cache.get(run_path)
            if metrics is not None:
                raw.update(metrics.metric_names)
        return raw

    def current_x_axis(self) -> str:
        axes: list[str] = []
        for run_path in self.selected_run_paths:
            metrics = self.metrics_cache.get(run_path)
            if metrics is not None:
                axis = resolve_x_axis(metrics.frame, self.x_axis_mode)
                if axis not in axes:
                    axes.append(axis)
        return axes[0] if len(axes) == 1 else "x"

    def active_metric(self) -> str | None:
        if not self.selected_metrics:
            return None
        if self.active_metric_index >= len(self.selected_metrics):
            self.active_metric_index = 0
        return self.selected_metrics[self.active_metric_index]

    def run_by_path(self, run_path: str) -> RunVersion | None:
        for run in self.runs:
            if str(run.metrics_csv_path) == run_path:
                return run
        return None

    def persist_state(self) -> None:
        save_state(
            self.root,
            UiState(
                selected_run_paths=tuple(self.selected_run_paths),
                selected_metrics=tuple(self.selected_metrics),
                active_metric=self.active_metric(),
                grouped_mode=self.grouped_mode,
                x_axis_mode=self.x_axis_mode,
                dark_mode=self.dark_mode,
                smoothing=self.smoothing,
                log_x=self.log_x,
                log_y=self.log_y,
            ),
        )

    def action_open_metric_selector(self) -> None:
        choices = self.metric_choices_for_runs(self.selected_run_paths)
        items = [SelectorItem(choice, choice) for choice in choices]
        self.push_screen(SelectorScreen("Metrics", items, self.selected_metrics), self.apply_metric_selection)

    def apply_metric_selection(self, selected: list[str] | None) -> None:
        if selected is None:
            return
        choices = set(self.metric_choices_for_runs(self.selected_run_paths))
        self.selected_metrics = [metric for metric in selected if metric in choices]
        if not self.selected_metrics and choices:
            self.selected_metrics = [self.default_metric_choice(tuple(choices))]
        self.active_metric_index = 0
        self.render_current()

    def action_open_run_selector(self) -> None:
        items = [
            SelectorItem(
                str(run.metrics_csv_path),
                f"{run.display_name}  {run.status}",
                f"{run.display_name} {run.metrics_csv_path}",
            )
            for run in self.runs
        ]
        self.push_screen(SelectorScreen("Runs", items, self.selected_run_paths), self.apply_run_selection)

    def action_open_config_selector(self) -> None:
        items = [ConfigItem(run) for run in self.runs if run.config_yaml_path is not None]
        if not items:
            self.status_message = "no unique yaml configs found"
            self.render_current()
            return
        self.push_screen(ConfigScreen(items))

    def apply_run_selection(self, selected: list[str] | None) -> None:
        if selected is None:
            return
        self.selected_run_paths = [path for path in selected if path in self.metrics_cache]
        if not self.selected_run_paths and self.runs:
            latest = max(self.runs, key=lambda run: run.last_modified)
            self.selected_run_paths = [str(latest.metrics_csv_path)]
        choices = self.metric_choices_for_runs(self.selected_run_paths)
        self.selected_metrics = [metric for metric in self.selected_metrics if metric in choices]
        if not self.selected_metrics and choices:
            self.selected_metrics = [self.default_metric_choice(choices)]
        self.active_metric_index = 0
        self.render_current()

    def action_next_metric(self) -> None:
        if self.selected_metrics:
            self.active_metric_index = (self.active_metric_index + 1) % len(self.selected_metrics)
            self.render_current()

    def action_previous_metric(self) -> None:
        if self.selected_metrics:
            self.active_metric_index = (self.active_metric_index - 1) % len(self.selected_metrics)
            self.render_current()

    def action_toggle_x_axis(self) -> None:
        self.x_axis_mode = "epoch" if self.x_axis_mode == "step" else "step"
        self.status_message = f"x-axis: {self.x_axis_mode}"
        self.render_current()

    def action_toggle_dark_mode(self) -> None:
        self.dark_mode = not self.dark_mode
        self.status_message = f"theme: {theme_name(self.dark_mode)}"
        self.render_current()

    def action_toggle_smoothing(self) -> None:
        self.smoothing = not self.smoothing
        self.render_current()

    def action_toggle_log_x(self) -> None:
        self.log_x = not self.log_x
        self.render_current()

    def action_toggle_log_y(self) -> None:
        self.log_y = not self.log_y
        self.render_current()

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        self.status_message = "paused" if self.paused else "live"
        self.render_current()


def build_snapshot(root: Path) -> tuple[list[RunVersion], dict[str, RunMetrics]]:
    runs = discover_runs(root)
    metrics_cache: dict[str, RunMetrics] = {}
    updated_runs: list[RunVersion] = []
    for run in runs:
        metrics = load_run_metrics(run.metrics_csv_path)
        metrics_cache[str(run.metrics_csv_path)] = metrics
        updated_runs.append(replace(run, available_numeric_metrics=metrics.metric_names))
    return updated_runs, metrics_cache


def onoff(value: bool) -> str:
    return "on" if value else "off"


def theme_name(dark_mode: bool) -> str:
    return "dark" if dark_mode else "light"


def keybinding_bar() -> Text:
    items = (
        ("m", "metrics"),
        ("r", "runs"),
        ("c", "configs"),
        ("n/p", "metric"),
        ("a", "axis"),
        ("d", "theme"),
        ("s", "smooth"),
        ("x/y", "log"),
        ("sp", "pause"),
        ("R", "scan"),
        ("q", "quit"),
    )
    text = Text()
    for index, (key, label) in enumerate(items):
        if index:
            text.append("  ", style="dim")
        text.append(f" {key} ", style="bold reverse")
        text.append(f" {label}")
    return text
