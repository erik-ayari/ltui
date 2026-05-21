from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal


RunStatus = Literal["active", "stale", "finished"]


@dataclass(frozen=True)
class RunVersion:
    display_name: str
    metrics_csv_path: Path
    config_yaml_path: Path | None
    parent_name: str | None
    version_name: str | None
    last_modified: float
    status: RunStatus
    available_numeric_metrics: tuple[str, ...] = ()


def discover_runs(
    root: str | Path,
    *,
    now: float | None = None,
    active_age_seconds: float = 120,
    stale_age_seconds: float = 3600,
) -> list[RunVersion]:
    root_path = Path(root).expanduser().resolve()
    if now is None:
        now = time.time()

    runs: list[RunVersion] = []
    for metrics_path in sorted(root_path.rglob("metrics.csv")):
        if not metrics_path.is_file():
            continue
        stat = metrics_path.stat()
        parent_name, version_name, display_name = infer_run_names(root_path, metrics_path)
        runs.append(
            RunVersion(
                display_name=display_name,
                metrics_csv_path=metrics_path.resolve(),
                config_yaml_path=find_unique_config_yaml(metrics_path),
                parent_name=parent_name,
                version_name=version_name,
                last_modified=stat.st_mtime,
                status=classify_status(now - stat.st_mtime, active_age_seconds, stale_age_seconds),
            )
        )
    return runs


def infer_run_names(root: Path, metrics_path: Path) -> tuple[str | None, str | None, str]:
    rel_parts = metrics_path.resolve().relative_to(root.resolve()).parts[:-1]
    if not rel_parts:
        return None, None, metrics_path.parent.name

    version_name = rel_parts[-1]
    if len(rel_parts) >= 2 and rel_parts[-2] == "lightning_logs":
        parent_parts = rel_parts[:-2]
        parent_name = "/".join(parent_parts) if parent_parts else "lightning_logs"
        display_name = f"{parent_name}/{version_name}"
        return parent_name, version_name, display_name

    parent_name = "/".join(rel_parts[:-1]) if len(rel_parts) > 1 else None
    display_name = "/".join(rel_parts)
    return parent_name, version_name, display_name


def classify_status(age_seconds: float, active_age_seconds: float, stale_age_seconds: float) -> RunStatus:
    if age_seconds <= active_age_seconds:
        return "active"
    if age_seconds <= stale_age_seconds:
        return "stale"
    return "finished"


def find_unique_config_yaml(metrics_path: Path) -> Path | None:
    for directory in config_candidate_dirs(metrics_path):
        candidates = sorted(path.resolve() for path in directory.glob("*.y*ml") if path.is_file())
        if len(candidates) == 1:
            return candidates[0]
    return None


def config_candidate_dirs(metrics_path: Path) -> tuple[Path, ...]:
    version_dir = metrics_path.parent
    candidates = [version_dir]
    if version_dir.name.startswith("version_") and version_dir.parent.name in {"csv", "lightning_logs"}:
        sibling_version_dir = version_dir.parent.parent / version_dir.name
        if sibling_version_dir != version_dir:
            candidates.append(sibling_version_dir)
    return tuple(candidates)
