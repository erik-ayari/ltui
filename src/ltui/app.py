from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from rapidfuzz import fuzz
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from .data import AxisMode, ImageSource, RunMetrics, load_run_metrics, metric_series, resolve_family, resolve_x_axis
from .discovery import RunVersion, discover_runs
from .plotting import GridLayout, GridPage, PlotCurve, PlotPanel, render_plot, render_plot_grid
from .state import UiState, load_state, save_state, valid_state


PALETTE = ("blue+", "orange+", "cyan+", "magenta+", "green+", "red+", "white", "gray+")
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PlotSettings:
    x_axis_mode: AxisMode = "step"
    smoothing: bool = False
    log_x: bool = False
    log_y: bool = False


class SelectorItem:
    def __init__(
        self,
        key: str,
        label: str,
        search_text: str | None = None,
        selection_keys: tuple[str, ...] | None = None,
    ) -> None:
        self.key = key
        self.label = label
        self.search_text = search_text or label
        self.selection_keys = selection_keys or (key,)


class SelectorScreen(ModalScreen[list[str] | None]):
    BINDINGS = [
        Binding("/", "focus_search", "Search", show=False),
        Binding("space", "toggle", "Toggle", show=False),
        Binding("left", "previous_model", "Previous model", show=False),
        Binding("right", "next_model", "Next model", show=False),
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
        item = self.item_by_key(key)
        if item is None:
            return
        if all(selection_key in self.selected for selection_key in item.selection_keys):
            for selection_key in item.selection_keys:
                if selection_key in self.selected:
                    self.selected.remove(selection_key)
        else:
            for selection_key in item.selection_keys:
                if selection_key not in self.selected:
                    self.selected.append(selection_key)
        self.refresh_options(highlight_key=key)

    def action_apply(self) -> None:
        option_list = self.query_one(OptionList)
        if not self.selected and option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            if option.id is not None:
                item = self.item_by_key(str(option.id))
                if item is not None:
                    self.selected.extend(item.selection_keys)
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

    def action_previous_model(self) -> None:
        self.jump_model(-1)

    def action_next_model(self) -> None:
        self.jump_model(1)

    def jump_model(self, direction: int) -> None:
        option_list = self.query_one(OptionList)
        if option_list.highlighted is None:
            return
        model_indices = [index for index, item in enumerate(self.filtered_items()) if item.key.startswith("model:")]
        if direction > 0:
            candidates = [index for index in model_indices if index > option_list.highlighted]
            if candidates:
                option_list.highlighted = candidates[0]
        else:
            candidates = [index for index in model_indices if index < option_list.highlighted]
            if candidates:
                option_list.highlighted = candidates[-1]

    def filtered_items(self) -> list[SelectorItem]:
        if not self.search_query:
            return self.items
        scored = [
            (fuzz.WRatio(self.search_query, item.search_text), index, item)
            for index, item in enumerate(self.items)
        ]
        return [item for score, index, item in sorted(scored, key=lambda value: (-value[0], value[1])) if score >= 35]

    def format_label(self, item: SelectorItem) -> str:
        selected_count = sum(selection_key in self.selected for selection_key in item.selection_keys)
        if selected_count == len(item.selection_keys):
            marker = "[x]"
        elif selected_count:
            marker = "[-]"
        else:
            marker = "[ ]"
        return f"{marker} {item.label}"

    def item_by_key(self, key: str) -> SelectorItem | None:
        return next((item for item in self.items if item.key == key), None)


class ConfigItem:
    def __init__(self, key: str, label: str, runs: tuple[RunVersion, ...], search_text: str | None = None) -> None:
        self.key = key
        self.label = label
        self.runs = runs
        self.search_text = search_text or " ".join(
            f"{run.display_name} {run.metrics_csv_path} {run.config_yaml_path}" for run in runs
        )


class ConfigScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("/", "focus_search", "Search", show=False),
        Binding("left", "previous_model", "Previous model", show=False),
        Binding("right", "next_model", "Next model", show=False),
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

    def action_previous_model(self) -> None:
        self.jump_model(-1)

    def action_next_model(self) -> None:
        self.jump_model(1)

    def jump_model(self, direction: int) -> None:
        option_list = self.query_one(OptionList)
        if option_list.highlighted is None:
            return
        model_indices = [index for index, item in enumerate(self.filtered_items()) if item.key.startswith("model:")]
        if direction > 0:
            candidates = [index for index in model_indices if index > option_list.highlighted]
            if candidates:
                option_list.highlighted = candidates[0]
                self.update_preview()
        else:
            candidates = [index for index in model_indices if index < option_list.highlighted]
            if candidates:
                option_list.highlighted = candidates[-1]
                self.update_preview()

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
        if selected is None:
            text_area.load_text("No config selected.")
            return
        text_area.load_text(config_preview_text(selected.runs))

    def selected_item(self) -> ConfigItem | None:
        option_list = self.query_one(OptionList)
        if option_list.highlighted is None:
            return None
        option = option_list.get_option_at_index(option_list.highlighted)
        if option.id is None:
            return None
        key = str(option.id)
        return next((item for item in self.items if item.key == key), None)


class SingleSelectorItem:
    def __init__(self, key: str, label: str, search_text: str | None = None, selectable: bool = True) -> None:
        self.key = key
        self.label = label
        self.search_text = search_text or label
        self.selectable = selectable


class SingleSelectorScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("/", "focus_search", "Search", show=False),
        Binding("enter", "apply", "Apply", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, title: str, items: list[SingleSelectorItem]) -> None:
        super().__init__()
        self.title = title
        self.items = items
        self.search_query = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="selector"):
            yield Static(self.title, id="selector-title")
            yield Input(placeholder="Search", id="search")
            yield OptionList(id="options", markup=False)
            yield Static("/ search  arrows navigate  enter open  escape cancel", id="selector-help")

    def on_mount(self) -> None:
        self.refresh_options()
        self.query_one(OptionList).focus()

    def action_focus_search(self) -> None:
        self.query_one(Input).focus()

    def action_apply(self) -> None:
        item = self.selected_item()
        if item is not None and item.selectable:
            self.dismiss(item.key)

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

    def refresh_options(self) -> None:
        option_list = self.query_one(OptionList)
        option_list.clear_options()
        option_list.add_options(Option(item.label, id=item.key) for item in self.filtered_items())
        if option_list.option_count:
            option_list.highlighted = 0

    def filtered_items(self) -> list[SingleSelectorItem]:
        if not self.search_query:
            return self.items
        scored = [
            (fuzz.WRatio(self.search_query, item.search_text), index, item)
            for index, item in enumerate(self.items)
        ]
        return [item for score, index, item in sorted(scored, key=lambda value: (-value[0], value[1])) if score >= 35]

    def selected_item(self) -> SingleSelectorItem | None:
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
        background: black;
        color: white;
    }

    #header {
        height: 3;
        padding: 0 1;
        background: black;
        color: white;
    }

    #plot {
        height: 1fr;
        padding: 0 1;
        background: black;
    }

    #footer {
        height: auto;
        min-height: 1;
        max-height: 2;
        padding: 0 1;
        background: black;
        color: gray;
    }

    SelectorScreen {
        align: center middle;
    }

    ConfigScreen {
        align: center middle;
    }

    SingleSelectorScreen {
        align: center middle;
    }

    #selector {
        width: 82%;
        height: 82%;
        border: solid gray;
        background: black;
        color: white;
    }

    #selector-title {
        height: 1;
        padding: 0 1;
        color: white;
        text-style: bold;
    }

    #search {
        margin: 0 1;
        background: black;
        color: white;
    }

    #options {
        height: 1fr;
        margin: 0 1;
        background: black;
        color: white;
    }

    #selector-help {
        height: 1;
        padding: 0 1;
        color: gray;
    }

    #config-viewer {
        width: 88%;
        height: 86%;
        border: solid gray;
        background: black;
        color: white;
    }

    #config-title {
        height: 1;
        padding: 0 1;
        color: white;
        text-style: bold;
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
        background: black;
        color: white;
    }

    #config-options {
        height: 1fr;
        background: black;
        color: white;
    }

    #config-preview {
        width: 1fr;
        height: 1fr;
        border-left: solid gray;
        background: black;
        color: white;
    }

    #config-help {
        height: 1;
        padding: 0 1;
        color: gray;
    }
    """

    BINDINGS = [
        Binding("m", "open_metric_selector", "Metrics"),
        Binding("i", "open_image_selector", "Images"),
        Binding("r", "open_run_selector", "Runs"),
        Binding("c", "open_config_selector", "Configs"),
        Binding("n", "next_metric", "Next metric"),
        Binding("p", "previous_metric", "Previous metric"),
        Binding("a", "toggle_x_axis", "X axis"),
        Binding("d", "toggle_dark_mode", "Dark"),
        Binding("s", "toggle_smoothing", "Smoothing"),
        Binding("x", "toggle_log_x", "Log x"),
        Binding("y", "toggle_log_y", "Log y"),
        Binding("space", "open_multiplot", "Multiplot"),
        Binding("left", "multiplot_left", "Left", show=False),
        Binding("right", "multiplot_right", "Right", show=False),
        Binding("up", "multiplot_up", "Up", show=False),
        Binding("down", "multiplot_down", "Down", show=False),
        Binding("enter", "focus_multiplot", "Focus", show=False),
        Binding("escape", "clear_multiplot_selection", "Clear", show=False),
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
        self.multiplot = False
        self.multiplot_selection: int | None = None
        self.multiplot_page = 0
        self.multiplot_layout = GridLayout(columns=1, rows=1, page_size=1, page_count=1)
        self.metric_settings: dict[str, PlotSettings] = {}
        self.status_message = "starting"
        self.refreshing = False

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        yield Static("", id="plot")
        yield Static("", id="footer")

    def on_mount(self) -> None:
        self.update_footer()
        asyncio.create_task(self.refresh_snapshot(initial=True))
        self.set_interval(1.5, self.schedule_live_refresh)

    def schedule_live_refresh(self) -> None:
        if not self.refreshing:
            asyncio.create_task(self.refresh_snapshot())

    async def refresh_snapshot(self, *, initial: bool = False) -> None:
        if self.refreshing:
            return
        self.refreshing = True
        try:
            runs, metrics_cache = await asyncio.to_thread(build_snapshot, self.root)
            self.apply_snapshot(runs, metrics_cache, initial)
            self.status_message = "live"
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
        restore_multiplot = False
        if not runs:
            self.selected_run_paths = []
            self.selected_metrics = []
            self.active_metric_index = 0
            self.multiplot = False
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
                restore_multiplot = restored.multiplot

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
        self.metric_settings = {metric: settings for metric, settings in self.metric_settings.items() if metric in self.selected_metrics}
        if initial:
            self.multiplot = restore_multiplot and bool(self.selected_metrics)
            if not self.multiplot:
                self.multiplot_selection = None
                self.multiplot_page = 0
        self.clamp_multiplot_selection()

    def render_current(self) -> None:
        plot = self.query_one("#plot", Static)
        if not self.runs:
            self.update_header()
            plot.update(self.empty_text())
            self.update_footer()
            return
        if not self.selected_metrics:
            self.update_header()
            plot.update("No numeric metrics found in selected run.")
            self.update_footer()
            return

        size = plot.size
        if self.multiplot:
            self.render_multiplot(plot, size)
        else:
            self.render_focused_plot(plot, size)
        self.persist_state()

    def render_focused_plot(self, plot: Static, size) -> None:
        active_metric = self.active_metric()
        if active_metric is None:
            plot.update("No numeric metrics found in selected run.")
            return
        settings = self.settings_for_metric(active_metric)
        x_axis = self.current_x_axis(settings)
        result = render_plot(
            self.build_curves(active_metric, settings),
            width=max(size.width - 2, 60),
            height=max(size.height - 2, 16),
            title=active_metric,
            x_label=x_axis,
            y_label="",
            smoothing=settings.smoothing,
            log_x=settings.log_x,
            log_y=settings.log_y,
            x_min=0 if x_axis in {"step", "epoch"} else None,
            dark_mode=self.dark_mode,
        )
        if result.status_messages:
            self.status_message = " | ".join(result.status_messages)
        elif self.status_message.startswith("log-"):
            self.status_message = "live"
        self.update_header()
        plot.update(Text.from_ansi(result.text))
        self.update_footer()

    def render_multiplot(self, plot: Static, size) -> None:
        panels = [self.plot_panel(metric) for metric in self.selected_metrics]
        result = render_plot_grid(
            panels,
            width=max(size.width - 2, 60),
            height=max(size.height - 2, 16),
            page=self.multiplot_page,
            selected_index=self.multiplot_selection,
        )
        self.multiplot_layout = result.layout
        self.multiplot_page = result.page
        if result.status_messages:
            self.status_message = " | ".join(result.status_messages)
        elif self.status_message.startswith("log-"):
            self.status_message = "live"
        self.update_header()
        plot.update(Text.from_ansi(result.text))
        self.update_footer()

    def plot_panel(self, metric: str) -> PlotPanel:
        settings = self.settings_for_metric(metric)
        x_axis = self.current_x_axis(settings)
        group_title, title = metric_plot_titles(metric)
        return PlotPanel(
            title=title,
            curves=self.build_curves(metric, settings),
            x_label=x_axis,
            group_title=group_title,
            smoothing=settings.smoothing,
            log_x=settings.log_x,
            log_y=settings.log_y,
            x_min=0 if x_axis in {"step", "epoch"} else None,
            dark_mode=self.dark_mode,
        )

    def update_header(self) -> None:
        header = self.query_one("#header", Static)
        header.update(
            header_bar(self.data_summary_text(), self.config_count(), self.total_metric_count())
        )

    def data_summary_text(self) -> Text:
        groups = group_runs_by_model(self.runs)
        if not groups:
            return Text("none")

        text = Text()
        for group_index, (model, runs) in enumerate(groups):
            if group_index:
                text.append(", ")
            text.append(model)
            text.append(" (")
            self.append_version_summary(text, runs)
            text.append(")")
        return text

    def append_version_summary(self, text: Text, runs: list[RunVersion]) -> None:
        visible_runs = runs if len(runs) <= 3 else runs[:2]
        for index, run in enumerate(visible_runs):
            if index:
                text.append(", ")
            text.append(run_version_label(run), style="green" if run.status == "active" else "")
        if len(runs) > 3:
            text.append(f" and {len(runs) - 2} more")

    def config_count(self) -> int:
        return len({run.config_yaml_path for run in self.runs if run.config_yaml_path is not None})

    def total_metric_count(self) -> int:
        return len(self.metric_choices_for_runs([str(run.metrics_csv_path) for run in self.runs]))

    def empty_text(self) -> str:
        return (
            f"No metrics.csv files found under {self.root}\n\n"
            "Expected examples:\n"
            "  lightning_logs/version_0/metrics.csv\n"
            "  run_a/version_0/metrics.csv\n"
            "  run_a/lightning_logs/version_0/metrics.csv\n\n"
            "Live discovery runs automatically. Press q to quit."
        )

    def update_footer(self) -> None:
        footer = self.query_one("#footer", Static)
        footer.update(keybinding_bar(self.page_indicator(), widget_width(footer), self.multiplot))

    def page_indicator(self) -> str | None:
        if not self.multiplot or self.multiplot_layout.page_count <= 1:
            return None
        labels = []
        for page in range(self.multiplot_layout.page_count):
            label = str(page + 1)
            labels.append(f"[{label}]" if page == self.multiplot_page else label)
        return "pages " + " ".join(labels)

    def build_curves(self, metric: str | None = None, settings: PlotSettings | None = None) -> list[PlotCurve]:
        curves: list[PlotCurve] = []
        metric = metric or self.active_metric()
        if metric is None:
            return curves
        settings = settings or self.settings_for_metric(metric)

        for run_index, run_path in enumerate(self.selected_run_paths):
            metrics = self.metrics_cache.get(run_path)
            run = self.run_by_path(run_path)
            if metrics is None or run is None:
                continue
            color = PALETTE[run_index % len(PALETTE)]
            metric_roles = (
                resolve_family(metrics, metric, settings.x_axis_mode)
                if self.grouped_mode
                else ((metric, "raw"),)
            )
            for metric_name, role in metric_roles:
                series = metric_series(metrics, metric_name, settings.x_axis_mode)
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
                        run_label=run_display_name(run),
                        run_status=run.status,
                        style_label=self.style_label(metric_name, role),
                    )
                )
        return curves

    def curve_label(self, run: RunVersion, metric_name: str, role: str) -> str:
        style_label = self.style_label(metric_name, role)
        if self.grouped_mode and role in {"train", "val"}:
            return style_label if len(self.selected_run_paths) == 1 else f"{run_display_name(run)} {style_label}"
        return metric_name if len(self.selected_run_paths) == 1 else f"{run_display_name(run)} {metric_name}"

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

    def image_choices_for_runs(self, run_paths: list[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        choices: list[str] = []
        for run_path in run_paths:
            metrics = self.metrics_cache.get(run_path)
            if metrics is None:
                continue
            for source in metrics.image_sources:
                if source.name not in seen:
                    seen.add(source.name)
                    choices.append(source.name)
        return tuple(choices)

    def image_runs_for_choice(self, image_name: str) -> list[tuple[RunVersion, ImageSource]]:
        candidates: list[tuple[RunVersion, ImageSource]] = []
        for run_path in self.selected_run_paths:
            metrics = self.metrics_cache.get(run_path)
            run = self.run_by_path(run_path)
            if metrics is None or run is None:
                continue
            for source in metrics.image_sources:
                if source.name == image_name:
                    candidates.append((run, source))
        return candidates

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

    def current_x_axis(self, settings: PlotSettings | None = None) -> str:
        settings = settings or self.global_settings()
        axes: list[str] = []
        for run_path in self.selected_run_paths:
            metrics = self.metrics_cache.get(run_path)
            if metrics is not None:
                axis = settings.x_axis_mode if metrics.structured_sources else resolve_x_axis(metrics.frame, settings.x_axis_mode)
                if axis not in axes:
                    axes.append(axis)
        return axes[0] if len(axes) == 1 else "x"

    def global_settings(self) -> PlotSettings:
        return PlotSettings(
            x_axis_mode=self.x_axis_mode,
            smoothing=self.smoothing,
            log_x=self.log_x,
            log_y=self.log_y,
        )

    def settings_for_metric(self, metric: str) -> PlotSettings:
        return self.metric_settings.get(metric, self.global_settings())

    def selected_multiplot_metric(self) -> str | None:
        if self.multiplot_selection is None:
            return None
        if 0 <= self.multiplot_selection < len(self.selected_metrics):
            return self.selected_metrics[self.multiplot_selection]
        return None

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
                multiplot=self.multiplot,
            ),
        )

    def action_open_metric_selector(self) -> None:
        choices = self.metric_choices_for_runs(self.selected_run_paths)
        items = metric_selector_items(choices)
        self.push_screen(SelectorScreen("Metrics", items, self.selected_metrics), self.apply_metric_selection)

    def action_open_image_selector(self) -> None:
        choices = self.image_choices_for_runs(self.selected_run_paths)
        if not choices:
            self.status_message = "no logged images found"
            self.render_current()
            return
        self.push_screen(SingleSelectorScreen("Images", image_selector_items(choices)), self.apply_image_selection)

    def apply_image_selection(self, image_name: str | None) -> None:
        if image_name is None:
            return
        candidates = self.image_runs_for_choice(image_name)
        if not candidates:
            self.status_message = f"no images found: {image_name}"
            self.render_current()
            return
        if len(candidates) == 1:
            run, source = candidates[0]
            self.open_image_source(run, source)
            return

        items = [
            SingleSelectorItem(
                str(run.metrics_csv_path),
                run_display_name(run),
                f"{run_display_name(run)} {run.display_name} {run.metrics_csv_path}",
            )
            for run, source in candidates
        ]
        self.push_screen(
            SingleSelectorScreen("Image Run", items),
            lambda selected: self.apply_image_run_selection(image_name, selected),
        )

    def apply_image_run_selection(self, image_name: str, run_path: str | None) -> None:
        if run_path is None:
            return
        for run, source in self.image_runs_for_choice(image_name):
            if str(run.metrics_csv_path) == run_path:
                self.open_image_source(run, source)
                return
        self.status_message = f"no images found: {image_name}"
        self.render_current()

    def open_image_source(self, run: RunVersion, source: ImageSource) -> None:
        feh = shutil.which("feh")
        if feh is None:
            self.status_message = "feh not found"
            self.render_current()
            return
        if not source.directory.is_dir() or not image_files(source.directory):
            self.status_message = f"no image files: {source.name}"
            self.render_current()
            return
        subprocess.Popen(
            [feh, "--sort", "filename", str(source.directory)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.status_message = f"opened images: {run_display_name(run)} {source.name}"
        self.render_current()

    def apply_metric_selection(self, selected: list[str] | None) -> None:
        if selected is None:
            return
        choices = set(self.metric_choices_for_runs(self.selected_run_paths))
        self.selected_metrics = [metric for metric in selected if metric in choices]
        if not self.selected_metrics and choices:
            self.selected_metrics = [self.default_metric_choice(tuple(choices))]
        self.active_metric_index = 0
        self.metric_settings = {metric: settings for metric, settings in self.metric_settings.items() if metric in self.selected_metrics}
        self.multiplot_selection = None
        self.multiplot_page = 0
        self.render_current()

    def action_open_run_selector(self) -> None:
        items = run_selector_items(self.runs)
        self.push_screen(SelectorScreen("Runs", items, self.selected_run_paths), self.apply_run_selection)

    def action_open_config_selector(self) -> None:
        items = config_items(self.runs)
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
        self.metric_settings = {metric: settings for metric, settings in self.metric_settings.items() if metric in self.selected_metrics}
        self.multiplot_selection = None
        self.multiplot_page = 0
        self.render_current()

    def action_next_metric(self) -> None:
        if self.multiplot:
            self.move_multiplot_page(1)
            return
        if self.selected_metrics:
            self.active_metric_index = (self.active_metric_index + 1) % len(self.selected_metrics)
            self.render_current()

    def action_previous_metric(self) -> None:
        if self.multiplot:
            self.move_multiplot_page(-1)
            return
        if self.selected_metrics:
            self.active_metric_index = (self.active_metric_index - 1) % len(self.selected_metrics)
            self.render_current()

    def action_toggle_x_axis(self) -> None:
        metric = self.selected_multiplot_metric()
        if metric is None:
            self.x_axis_mode = "epoch" if self.x_axis_mode == "step" else "step"
            self.metric_settings.clear()
            self.status_message = f"x-axis: {self.x_axis_mode}"
        else:
            settings = self.settings_for_metric(metric)
            next_axis = "epoch" if settings.x_axis_mode == "step" else "step"
            self.metric_settings[metric] = replace(settings, x_axis_mode=next_axis)
            self.status_message = f"{metric}: x-axis {next_axis}"
        self.render_current()

    def action_toggle_dark_mode(self) -> None:
        self.dark_mode = not self.dark_mode
        self.status_message = f"theme: {theme_name(self.dark_mode)}"
        self.render_current()

    def action_toggle_smoothing(self) -> None:
        metric = self.selected_multiplot_metric()
        if metric is None:
            self.smoothing = not self.smoothing
            self.metric_settings.clear()
        else:
            settings = self.settings_for_metric(metric)
            self.metric_settings[metric] = replace(settings, smoothing=not settings.smoothing)
            self.status_message = f"{metric}: smooth {onoff(not settings.smoothing)}"
        self.render_current()

    def action_toggle_log_x(self) -> None:
        metric = self.selected_multiplot_metric()
        if metric is None:
            self.log_x = not self.log_x
            self.metric_settings.clear()
        else:
            settings = self.settings_for_metric(metric)
            self.metric_settings[metric] = replace(settings, log_x=not settings.log_x)
            self.status_message = f"{metric}: log-x {onoff(not settings.log_x)}"
        self.render_current()

    def action_toggle_log_y(self) -> None:
        metric = self.selected_multiplot_metric()
        if metric is None:
            self.log_y = not self.log_y
            self.metric_settings.clear()
        else:
            settings = self.settings_for_metric(metric)
            self.metric_settings[metric] = replace(settings, log_y=not settings.log_y)
            self.status_message = f"{metric}: log-y {onoff(not settings.log_y)}"
        self.render_current()

    def action_open_multiplot(self) -> None:
        if not self.selected_metrics:
            return
        self.multiplot = True
        self.multiplot_selection = None
        self.multiplot_page = 0
        self.status_message = "multiplot"
        self.render_current()

    def action_multiplot_left(self) -> None:
        self.move_multiplot_selection(0, -1)

    def action_multiplot_right(self) -> None:
        self.move_multiplot_selection(0, 1)

    def action_multiplot_up(self) -> None:
        self.move_multiplot_selection(-1, 0)

    def action_multiplot_down(self) -> None:
        self.move_multiplot_selection(1, 0)

    def action_focus_multiplot(self) -> None:
        if not self.multiplot:
            return
        if self.multiplot_selection is None:
            return
        self.active_metric_index = self.multiplot_selection
        self.multiplot = False
        self.multiplot_selection = None
        self.status_message = f"focused: {self.active_metric()}"
        self.render_current()

    def action_clear_multiplot_selection(self) -> None:
        if not self.multiplot or self.multiplot_selection is None:
            return
        self.multiplot_selection = None
        self.render_current()

    def move_multiplot_selection(self, row_delta: int, column_delta: int) -> None:
        if not self.multiplot or not self.selected_metrics:
            return
        self.clamp_multiplot_selection()
        if self.multiplot_selection is None:
            self.multiplot_selection = self.first_index_on_multiplot_page()
            self.render_current()
            return

        if self.multiplot_layout.pages:
            self.move_grouped_multiplot_selection(row_delta, column_delta)
            return

        columns = self.multiplot_layout.columns
        page_size = self.multiplot_layout.page_size
        page_start = self.multiplot_page * page_size
        local = self.multiplot_selection - page_start
        row = local // columns
        column = local % columns
        next_index = page_start + (row + row_delta) * columns + column + column_delta

        if column_delta > 0 and (local == page_size - 1 or next_index >= min(page_start + page_size, len(self.selected_metrics))):
            next_index = min(page_start + page_size, len(self.selected_metrics) - 1)
        elif column_delta < 0 and local == 0:
            next_index = max(page_start - 1, 0)
        elif row_delta != 0 and not (page_start <= next_index < min(page_start + page_size, len(self.selected_metrics))):
            next_index = self.multiplot_selection

        self.multiplot_selection = min(max(next_index, 0), len(self.selected_metrics) - 1)
        self.multiplot_page = self.multiplot_selection // max(page_size, 1)
        self.render_current()

    def move_grouped_multiplot_selection(self, row_delta: int, column_delta: int) -> None:
        page = self.multiplot_layout.pages[self.multiplot_page]
        positions = multiplot_positions(page)
        current = positions.get(self.multiplot_selection)
        if current is None:
            self.multiplot_selection = page.panel_indices[0] if page.panel_indices else 0
            self.render_current()
            return

        if column_delta:
            self.multiplot_selection = self.horizontal_multiplot_target(page.panel_indices, column_delta)
            self.multiplot_page = self.page_for_multiplot_index(self.multiplot_selection)
            self.render_current()
            return

        target = vertical_multiplot_target(current, positions, row_delta)
        if target is not None:
            self.multiplot_selection = target
        self.render_current()

    def horizontal_multiplot_target(self, indices: tuple[int, ...], delta: int) -> int:
        if not indices or self.multiplot_selection is None:
            return 0
        local = indices.index(self.multiplot_selection) if self.multiplot_selection in indices else 0
        next_local = local + delta
        if 0 <= next_local < len(indices):
            return indices[next_local]
        next_page = self.multiplot_page + (1 if delta > 0 else -1)
        if 0 <= next_page < len(self.multiplot_layout.pages):
            page_indices = self.multiplot_layout.pages[next_page].panel_indices
            if page_indices:
                return page_indices[0 if delta > 0 else -1]
        return self.multiplot_selection

    def move_multiplot_page(self, delta: int) -> None:
        if not self.multiplot or self.multiplot_layout.page_count <= 1:
            return

        if self.multiplot_layout.pages:
            self.move_grouped_multiplot_page(delta)
            return

        page_size = max(self.multiplot_layout.page_size, 1)
        current_page = self.multiplot_page
        next_page = (current_page + delta) % self.multiplot_layout.page_count
        if self.multiplot_selection is not None:
            local = self.multiplot_selection - current_page * page_size
            page_start = next_page * page_size
            page_end = min(page_start + page_size, len(self.selected_metrics))
            self.multiplot_selection = min(page_start + max(local, 0), page_end - 1)
        self.multiplot_page = next_page
        self.render_current()

    def move_grouped_multiplot_page(self, delta: int) -> None:
        current_page = self.multiplot_page
        next_page = (current_page + delta) % self.multiplot_layout.page_count
        if self.multiplot_selection is not None:
            current_indices = self.multiplot_layout.pages[current_page].panel_indices
            next_indices = self.multiplot_layout.pages[next_page].panel_indices
            local = current_indices.index(self.multiplot_selection) if self.multiplot_selection in current_indices else 0
            if next_indices:
                self.multiplot_selection = next_indices[min(local, len(next_indices) - 1)]
        self.multiplot_page = next_page
        self.render_current()

    def clamp_multiplot_selection(self) -> None:
        if not self.selected_metrics:
            self.multiplot_selection = None
            self.multiplot_page = 0
            return
        if self.multiplot_selection is not None:
            self.multiplot_selection = min(self.multiplot_selection, len(self.selected_metrics) - 1)
            if self.multiplot_layout.pages:
                self.multiplot_page = self.page_for_multiplot_index(self.multiplot_selection)
                return
        self.multiplot_page = min(max(self.multiplot_page, 0), self.multiplot_layout.page_count - 1)

    def first_index_on_multiplot_page(self) -> int:
        if self.multiplot_layout.pages:
            indices = self.multiplot_layout.pages[self.multiplot_page].panel_indices
            return indices[0] if indices else 0
        return self.multiplot_page * self.multiplot_layout.page_size

    def page_for_multiplot_index(self, index: int) -> int:
        for page_index, page in enumerate(self.multiplot_layout.pages):
            if index in page.panel_indices:
                return page_index
        return min(max(self.multiplot_page, 0), self.multiplot_layout.page_count - 1)


def build_snapshot(root: Path) -> tuple[list[RunVersion], dict[str, RunMetrics]]:
    runs = discover_runs(root)
    metrics_cache: dict[str, RunMetrics] = {}
    updated_runs: list[RunVersion] = []
    for run in runs:
        metrics = load_run_metrics(run.metrics_csv_path)
        metrics_cache[str(run.metrics_csv_path)] = metrics
        updated_runs.append(replace(run, available_numeric_metrics=metrics.metric_names))
    return updated_runs, metrics_cache


class MetricSelectorNode:
    def __init__(self, name: str, path: tuple[str, ...] = ()) -> None:
        self.name = name
        self.path = path
        self.metric: str | None = None
        self.children: dict[str, MetricSelectorNode] = {}


@dataclass(frozen=True)
class MultiplotPosition:
    index: int
    section: int
    row: int
    column: int
    order: int


def metric_plot_titles(metric: str) -> tuple[str, str]:
    parts = tuple(part for part in metric.split("/") if part)
    if len(parts) <= 1:
        return "", metric
    return "/".join(parts[:-1]), parts[-1]


def multiplot_positions(page: GridPage) -> dict[int, MultiplotPosition]:
    positions: dict[int, MultiplotPosition] = {}
    order = 0
    for section_index, section in enumerate(page.sections):
        for local, index in enumerate(section.panel_indices):
            positions[index] = MultiplotPosition(
                index=index,
                section=section_index,
                row=local // section.columns,
                column=local % section.columns,
                order=order,
            )
            order += 1
    return positions


def vertical_multiplot_target(
    current: MultiplotPosition,
    positions: dict[int, MultiplotPosition],
    delta: int,
) -> int | None:
    if delta == 0:
        return current.index

    candidates = [
        position
        for position in positions.values()
        if position.section == current.section and position.row == current.row + delta
    ]
    if not candidates:
        candidates = [
            position
            for position in positions.values()
            if (position.order - current.order) * delta > 0
        ]
    if not candidates:
        return None
    return min(candidates, key=lambda position: (abs(position.column - current.column), abs(position.order - current.order))).index


def metric_selector_items(choices: tuple[str, ...]) -> list[SelectorItem]:
    if not any("/" in choice for choice in choices):
        return [SelectorItem(choice, choice) for choice in choices]

    root = MetricSelectorNode("")
    for choice in choices:
        node = root
        parts = tuple(part for part in choice.split("/") if part)
        for index, part in enumerate(parts):
            if part not in node.children:
                node.children[part] = MetricSelectorNode(part, parts[: index + 1])
            node = node.children[part]
        node.metric = choice

    items: list[SelectorItem] = []
    for child in root.children.values():
        append_metric_selector_items(child, items, 0)
    return items


def append_metric_selector_items(node: MetricSelectorNode, items: list[SelectorItem], depth: int) -> None:
    subtree = metric_subtree(node)
    path = "/".join(node.path)
    if node.children:
        label = f"{'  ' * depth}{node.name}"
        if node.metric is not None:
            label += "  [metric]"
        items.append(SelectorItem(f"metric-group:{path}", label, " ".join([path, *subtree]), subtree))
    elif node.metric is not None:
        items.append(SelectorItem(node.metric, f"{'  ' * depth}{node.name}", node.metric, (node.metric,)))

    for child in node.children.values():
        append_metric_selector_items(child, items, depth + 1)


def metric_subtree(node: MetricSelectorNode) -> tuple[str, ...]:
    metrics: list[str] = []
    if node.metric is not None:
        metrics.append(node.metric)
    for child in node.children.values():
        metrics.extend(metric_subtree(child))
    return tuple(metrics)


def image_selector_items(choices: tuple[str, ...]) -> list[SingleSelectorItem]:
    if not any("/" in choice for choice in choices):
        return [SingleSelectorItem(choice, choice) for choice in choices]

    root = MetricSelectorNode("")
    for choice in choices:
        node = root
        parts = tuple(part for part in choice.split("/") if part)
        for index, part in enumerate(parts):
            if part not in node.children:
                node.children[part] = MetricSelectorNode(part, parts[: index + 1])
            node = node.children[part]
        node.metric = choice

    items: list[SingleSelectorItem] = []
    for child in root.children.values():
        append_image_selector_items(child, items, 0)
    return items


def append_image_selector_items(node: MetricSelectorNode, items: list[SingleSelectorItem], depth: int) -> None:
    path = "/".join(node.path)
    if node.children:
        label = f"{'  ' * depth}{node.name}"
        if node.metric is not None:
            label += "  [image]"
            items.append(SingleSelectorItem(node.metric, label, path))
        else:
            items.append(SingleSelectorItem(f"image-group:{path}", label, path, selectable=False))
    elif node.metric is not None:
        items.append(SingleSelectorItem(node.metric, f"{'  ' * depth}{node.name}", node.metric))

    for child in node.children.values():
        append_image_selector_items(child, items, depth + 1)


def image_files(directory: Path) -> tuple[Path, ...]:
    try:
        return tuple(sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES))
    except OSError:
        return ()


def run_selector_items(runs: list[RunVersion]) -> list[SelectorItem]:
    items: list[SelectorItem] = []
    for model, group_runs in group_runs_by_model(runs):
        keys = tuple(str(run.metrics_csv_path) for run in group_runs)
        statuses = "/".join(sorted({run.status for run in group_runs}))
        labels = [run_display_name(run) for run in group_runs]
        items.append(
            SelectorItem(
                f"model:{model}",
                f"{model}  {statuses}",
                " ".join([model, *labels, *[run.display_name for run in group_runs], *keys]),
                keys,
            )
        )
        for run in group_runs:
            key = str(run.metrics_csv_path)
            version = run_version_label(run)
            search_text = f"{model} {version} {run_display_name(run)} {run.display_name} {key}"
            items.append(SelectorItem(key, f"  {version}  {run.status}", search_text))
    return items


def config_items(runs: list[RunVersion]) -> list[ConfigItem]:
    config_runs = [run for run in runs if run.config_yaml_path is not None]
    items: list[ConfigItem] = []
    for model, group_runs in group_runs_by_model(config_runs):
        search_text = " ".join(
            [
                model,
                *[run_display_name(run) for run in group_runs],
                *[run.display_name for run in group_runs],
                *[str(run.metrics_csv_path) for run in group_runs],
            ]
        )
        items.append(ConfigItem(f"model:{model}", model, tuple(group_runs), search_text))
        for run in group_runs:
            search_text = f"{run_display_name(run)} {run.display_name} {run.metrics_csv_path}"
            items.append(ConfigItem(str(run.metrics_csv_path), f"  {run_version_label(run)}", (run,), search_text))
    return items


def group_runs_by_model(runs: list[RunVersion]) -> list[tuple[str, list[RunVersion]]]:
    groups: list[tuple[str, list[RunVersion]]] = []
    by_model: dict[str, list[RunVersion]] = {}
    for run in runs:
        model = run_model_name(run)
        if model not in by_model:
            by_model[model] = []
            groups.append((model, by_model[model]))
        by_model[model].append(run)
    return groups


def run_model_name(run: RunVersion) -> str:
    return run.parent_name or abbreviate_version(run.display_name)


def run_version_label(run: RunVersion) -> str:
    return abbreviate_version(run.version_name or run.display_name)


def run_display_name(run: RunVersion) -> str:
    model = run_model_name(run)
    version = run_version_label(run)
    return f"{model}/{version}" if run.parent_name else version


def abbreviate_version(value: str) -> str:
    return "/".join(abbreviate_version_part(part) for part in value.split("/"))


def abbreviate_version_part(value: str) -> str:
    prefix = "version_"
    if value.startswith(prefix) and value[len(prefix):].isdigit():
        return f"v{value[len(prefix):]}"
    return value


def config_preview_text(runs: tuple[RunVersion, ...]) -> str:
    sections: list[str] = []
    for run in runs:
        if run.config_yaml_path is None:
            continue
        try:
            text = run.config_yaml_path.read_text(errors="replace")
        except OSError as exc:
            text = f"Could not read {run.config_yaml_path}\n\n{exc}"
        sections.append(f"# {run_display_name(run)}\n# {run.config_yaml_path}\n\n{text}")
    return "\n\n---\n\n".join(sections) if sections else "No config selected."


def onoff(value: bool) -> str:
    return "on" if value else "off"


def theme_name(dark_mode: bool) -> str:
    return "dark" if dark_mode else "light"


def header_bar(runs: Text, config_count: int, metric_count: int) -> Text:
    text = label_value_segment("runs", runs)
    text.append("\n")
    text.append(label_value_segment("configs", Text(str(config_count), style="bold")))
    text.append("\n")
    text.append(label_value_segment("metrics", Text(str(metric_count), style="bold")))
    return text


def label_value_segment(label: str, value: Text) -> Text:
    text = Text()
    text.append(label, style="bold")
    text.append(": ", style="dim white")
    text.append(value)
    return text


def widget_width(widget: Static) -> int:
    return max(widget.size.width - 2, 1)


def keybinding_bar(page_indicator: str | None = None, width: int | None = None, multiplot: bool = False) -> Text:
    items = (
        ("r", "runs"),
        ("c", "configs"),
        ("m", "metrics"),
        ("i", "images"),
        ("n/p", "page" if multiplot else "metric"),
        ("a", "axis"),
        ("s", "smooth"),
        ("x/y", "log"),
        ("sp", "multi"),
        ("ent", "focus"),
        ("esc", "clear"),
        ("d", "theme"),
        ("q", "quit"),
    )
    segments: list[Text] = []
    for key, label in items:
        segment = Text()
        segment.append("[", style="dim")
        segment.append(key, style="bold")
        segment.append("]", style="dim")
        segment.append(f" {label}")
        segments.append(segment)
    if page_indicator:
        segments.append(Text(page_indicator, style="white"))
    return distributed_bar(segments, width)


def distributed_bar(segments: list[Text], width: int | None = None) -> Text:
    if width is None or minimum_width(segments) <= width:
        return distributed_row(segments, width)

    text = Text()
    for row_index, row in enumerate(split_segments(segments, width)):
        if row_index:
            text.append("\n")
        text.append(distributed_row(row, width))
    return text


def distributed_row(segments: list[Text], width: int | None = None) -> Text:
    text = Text()
    if not segments:
        return text

    gaps = len(segments) - 1
    spaces = gap_sizes(segments, width) if gaps else ()
    for index, segment in enumerate(segments):
        if index:
            text.append(" " * spaces[index - 1], style="dim")
        text.append(segment.copy())
    return text


def gap_sizes(segments: list[Text], width: int | None) -> tuple[int, ...]:
    gaps = len(segments) - 1
    if not gaps:
        return ()
    if width is None:
        return tuple(2 for _ in range(gaps))

    content_width = sum(text_width(segment) for segment in segments)
    available = max(width - content_width, gaps * 2)
    base = available // gaps
    remainder = available % gaps
    return tuple(base + (1 if index < remainder else 0) for index in range(gaps))


def split_segments(segments: list[Text], width: int | None = None) -> list[list[Text]]:
    if len(segments) <= 1:
        return [segments]

    indices = range(1, len(segments))
    if width is not None:
        fitting_indices = [
            index for index in indices
            if minimum_width(segments[:index]) <= width and minimum_width(segments[index:]) <= width
        ]
        indices = fitting_indices or range(1, len(segments))

    split_index = min(
        indices,
        key=lambda index: abs(minimum_width(segments[:index]) - minimum_width(segments[index:])),
    )
    return [segments[:split_index], segments[split_index:]]


def minimum_width(segments: list[Text]) -> int:
    if not segments:
        return 0
    return sum(text_width(segment) for segment in segments) + (len(segments) - 1) * 2


def text_width(text: Text) -> int:
    return len(text.plain)
