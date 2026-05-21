from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .discovery import RunVersion


STATE_ROOT = Path.home() / ".local" / "state" / "ltui"
LEGACY_STATE_ROOT = Path.home() / ".local" / "state" / "lightning-tui"


@dataclass(frozen=True)
class UiState:
    selected_run_paths: tuple[str, ...] = ()
    selected_metrics: tuple[str, ...] = ()
    active_metric: str | None = None
    grouped_mode: bool = True
    x_axis_mode: str = "step"
    dark_mode: bool = True
    smoothing: bool = False
    log_x: bool = False
    log_y: bool = False


def state_path(root: str | Path) -> Path:
    return state_file(root, STATE_ROOT)


def legacy_state_path(root: str | Path) -> Path:
    return state_file(root, LEGACY_STATE_ROOT)


def state_file(root: str | Path, state_root: Path) -> Path:
    resolved = str(Path(root).expanduser().resolve())
    key = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:24]
    return state_root / f"{key}.json"


def load_state(root: str | Path) -> UiState | None:
    path = state_path(root)
    if not path.exists():
        path = legacy_state_path(root)
        if not path.exists():
            return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return UiState(
        selected_run_paths=tuple(data.get("selected_run_paths", ())),
        selected_metrics=tuple(data.get("selected_metrics", ())),
        active_metric=data.get("active_metric"),
        grouped_mode=bool(data.get("grouped_mode", True)),
        x_axis_mode=valid_x_axis_mode(data.get("x_axis_mode", "step")),
        dark_mode=bool(data.get("dark_mode", True)),
        smoothing=bool(data.get("smoothing", False)),
        log_x=bool(data.get("log_x", False)),
        log_y=bool(data.get("log_y", False)),
    )


def save_state(root: str | Path, state: UiState) -> None:
    path = state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    payload["selected_run_paths"] = list(state.selected_run_paths)
    payload["selected_metrics"] = list(state.selected_metrics)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def valid_state(saved: UiState, runs: list[RunVersion], metric_choices: tuple[str, ...]) -> UiState:
    run_paths = {str(run.metrics_csv_path) for run in runs}
    selected_runs = tuple(path for path in saved.selected_run_paths if path in run_paths)
    selected_metrics = tuple(metric for metric in saved.selected_metrics if metric in metric_choices)
    active_metric = saved.active_metric if saved.active_metric in selected_metrics else None
    if active_metric is None and selected_metrics:
        active_metric = selected_metrics[0]
    return UiState(
        selected_run_paths=selected_runs,
        selected_metrics=selected_metrics,
        active_metric=active_metric,
        grouped_mode=saved.grouped_mode,
        x_axis_mode=valid_x_axis_mode(saved.x_axis_mode),
        dark_mode=saved.dark_mode,
        smoothing=saved.smoothing,
        log_x=saved.log_x,
        log_y=saved.log_y,
    )


def valid_x_axis_mode(value: object) -> str:
    if value in {"step", "epoch"}:
        return str(value)
    return "step"
