from __future__ import annotations

from argparse import Namespace
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any


try:
    from lightning.pytorch.loggers.logger import Logger as LightningLogger
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
except ImportError:
    LightningLogger = object

    def rank_zero_only(function):
        return function


class StructuredMetricWriter:
    def __init__(
        self,
        log_dir: str | Path,
        *,
        separator: str = "/",
        train_prefix: str = "train",
        val_prefix: str = "val",
    ) -> None:
        if not separator:
            raise ValueError("separator must not be empty")
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.separator = separator
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.manifest_path = self.log_dir / "ltui_manifest.json"
        self.series: dict[str, dict[str, Any]] = {}
        self.images: dict[str, dict[str, Any]] = {}
        self.paths: dict[Path, str] = {}
        self.image_paths: dict[Path, str] = {}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.load_manifest()

    def log_metrics(self, metrics: dict[str, Any], step: int | float | None = None) -> None:
        epoch = scalar(metrics.get("epoch"))
        explicit_step = scalar(metrics.get("step"))
        resolved_step = scalar(step)
        if resolved_step is None:
            resolved_step = explicit_step
        wall_time = time.time()

        for name, value in metrics.items():
            if name in {"epoch", "step"}:
                continue
            numeric = scalar(value)
            if numeric is None:
                continue
            self.log_metric(name, numeric, resolved_step, epoch, wall_time)
        self.save_manifest()

    def log_metric(self, name: str, value: float, step: float | None, epoch: float | None, wall_time: float) -> None:
        entry = self.series_entry(name)
        path = self.log_dir / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(["step", "epoch", "wall_time", "value"])
            writer.writerow([empty_if_none(step), empty_if_none(epoch), wall_time, value])

    def log_images(self, images: dict[str, Any], step: int | float | None = None, epoch: int | float | None = None) -> None:
        for name, image in images.items():
            self.log_image(name, image, step=step, epoch=epoch)

    def log_image(
        self,
        name: str,
        image: Any,
        *,
        step: int | float | None = None,
        epoch: int | float | None = None,
        extension: str = "png",
        dataformats: str = "HWC",
        wall_time: float | None = None,
    ) -> Path:
        resolved_step = scalar(step)
        resolved_epoch = scalar(epoch)
        resolved_wall_time = time.time() if wall_time is None else wall_time
        entry = self.image_entry(name)
        directory = self.log_dir / entry["path"]
        directory.mkdir(parents=True, exist_ok=True)
        path = next_image_path(directory, resolved_step, resolved_epoch, resolved_wall_time, image_extension(image, extension))
        write_image_file(image, path, dataformats)
        self.save_manifest()
        return path

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: int | float | None = None,
        walltime: float | None = None,
        dataformats: str = "CHW",
    ) -> None:
        self.log_image(tag, img_tensor, step=global_step, dataformats=dataformats, wall_time=walltime)

    def series_entry(self, name: str) -> dict[str, Any]:
        if name in self.series:
            return self.series[name]

        role, role_name, metric_path = self.parse_metric_name(name)
        path = self.series_path(role_name, metric_path, name)
        entry = {
            "name": name,
            "role": role,
            "role_name": role_name,
            "metric_path": list(metric_path),
            "group": list(metric_path[:-1]),
            "metric": metric_path[-1],
            "path": path.as_posix(),
        }
        self.series[name] = entry
        self.paths[path] = name
        return entry

    def image_entry(self, name: str) -> dict[str, Any]:
        if name in self.images:
            return self.images[name]

        role, role_name, image_path = self.parse_metric_name(name)
        path = self.image_dir(role_name, image_path, name)
        entry = {
            "name": name,
            "role": role,
            "role_name": role_name,
            "image_path": list(image_path),
            "group": list(image_path[:-1]),
            "image": image_path[-1],
            "path": path.as_posix(),
        }
        self.images[name] = entry
        self.image_paths[path] = name
        return entry

    def parse_metric_name(self, name: str) -> tuple[str, str, tuple[str, ...]]:
        parts = tuple(part.strip() for part in name.split(self.separator) if part.strip())
        if not parts:
            return "raw", "raw", (name,)
        role_name = parts[0]
        if len(parts) >= 2 and role_name == self.train_prefix:
            return "train", role_name, parts[1:]
        if len(parts) >= 2 and role_name == self.val_prefix:
            return "val", role_name, parts[1:]
        train_tail = prefix_role_tail(role_name, self.train_prefix)
        if train_tail is not None:
            return "train", self.train_prefix, (train_tail, *parts[1:])
        val_tail = prefix_role_tail(role_name, self.val_prefix)
        if val_tail is not None:
            return "val", self.val_prefix, (val_tail, *parts[1:])
        return "raw", "raw", parts

    def series_path(self, role_name: str, metric_path: tuple[str, ...], name: str) -> Path:
        parts = [safe_path_part(role_name), *[safe_path_part(part) for part in metric_path]]
        path = Path("series", *parts).with_suffix(".csv")
        owner = self.paths.get(path)
        if owner is None or owner == name:
            return path
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        return path.with_name(f"{path.stem}-{digest}.csv")

    def image_dir(self, role_name: str, image_path: tuple[str, ...], name: str) -> Path:
        parts = [safe_path_part(role_name), *[safe_path_part(part) for part in image_path]]
        path = Path("images", *parts)
        owner = self.image_paths.get(path)
        if owner is None or owner == name:
            return path
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        return path.with_name(f"{path.name}-{digest}")

    def load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for item in manifest.get("series", ()):
            if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("path"), str):
                self.series[item["name"]] = item
                self.paths[Path(item["path"])] = item["name"]
        for item in manifest.get("images", ()):
            if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("path"), str):
                self.images[item["name"]] = item
                self.image_paths[Path(item["path"])] = item["name"]

    def save_manifest(self) -> None:
        payload = {
            "schema_version": 1,
            "separator": self.separator,
            "train_prefix": self.train_prefix,
            "val_prefix": self.val_prefix,
            "series": sorted(self.series.values(), key=lambda item: item["name"]),
            "images": sorted(self.images.values(), key=lambda item: item["name"]),
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class LtuiLogger(LightningLogger):
    def __init__(
        self,
        save_dir: str | Path,
        name: str = "ltui_logs",
        version: int | str | None = None,
        *,
        separator: str = "/",
        train_prefix: str = "train",
        val_prefix: str = "val",
    ) -> None:
        super().__init__()
        self.save_dir_path = Path(save_dir).expanduser().resolve()
        self.name_value = name
        self.version_value = resolve_version(self.save_dir_path / name, version)
        self.writer = StructuredMetricWriter(
            self.log_dir,
            separator=separator,
            train_prefix=train_prefix,
            val_prefix=val_prefix,
        )

    @property
    def name(self) -> str:
        return self.name_value

    @property
    def version(self) -> str:
        return self.version_value

    @property
    def save_dir(self) -> str:
        return str(self.save_dir_path)

    @property
    def log_dir(self) -> Path:
        return self.save_dir_path / self.name_value / self.version_value

    @property
    def experiment(self) -> StructuredMetricWriter:
        return self.writer

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.writer.log_metrics(metrics, step)

    @rank_zero_only
    def log_image(
        self,
        name: str,
        image: Any,
        *,
        step: int | float | None = None,
        epoch: int | float | None = None,
        extension: str = "png",
        dataformats: str = "HWC",
    ) -> Path:
        return self.writer.log_image(name, image, step=step, epoch=epoch, extension=extension, dataformats=dataformats)

    @rank_zero_only
    def log_images(self, images: dict[str, Any], step: int | float | None = None, epoch: int | float | None = None) -> None:
        self.writer.log_images(images, step=step, epoch=epoch)

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        path = self.log_dir / "hparams.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalize_params(params), indent=2, sort_keys=True, default=str) + "\n")

    @rank_zero_only
    def save(self) -> None:
        self.writer.save_manifest()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.writer.save_manifest()


def resolve_version(root: Path, version: int | str | None) -> str:
    if version is not None:
        return f"version_{version}" if isinstance(version, int) else str(version)
    if not root.exists():
        return "version_0"
    versions = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("version_") and path.name[8:].isdigit():
            versions.append(int(path.name[8:]))
    return f"version_{max(versions, default=-1) + 1}"


def scalar(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, ValueError):
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def empty_if_none(value: float | None) -> float | str:
    return "" if value is None else value


def next_image_path(directory: Path, step: float | None, epoch: float | None, wall_time: float, extension: str) -> Path:
    stem = f"{sort_part('step', step)}_{sort_part('epoch', epoch)}_time_{int(wall_time * 1000):013d}"
    path = directory / f"{stem}.{extension}"
    index = 1
    while path.exists():
        path = directory / f"{stem}_{index:04d}.{extension}"
        index += 1
    return path


def sort_part(label: str, value: float | None) -> str:
    if value is None:
        return f"{label}_none"
    if float(value).is_integer():
        return f"{label}_{int(value):012d}"
    return f"{label}_{value:020.6f}".replace("-", "neg")


def image_extension(image: Any, extension: str) -> str:
    if isinstance(image, (str, Path)):
        suffix = Path(image).suffix.lower().lstrip(".")
        if suffix:
            return suffix
    return extension.lower().lstrip(".") or "png"


def write_image_file(image: Any, path: Path, dataformats: str) -> None:
    if isinstance(image, (str, Path)) and Path(image).is_file():
        shutil.copyfile(Path(image), path)
        return
    if isinstance(image, (bytes, bytearray)):
        path.write_bytes(bytes(image))
        return
    if hasattr(image, "save"):
        image.save(path)
        return

    from PIL import Image
    import numpy as np

    value = image
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.asarray(value)
    if array.ndim == 3 and dataformats.upper() == "CHW":
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0, 1) * 255
    array = np.clip(array, 0, 255).astype("uint8")
    Image.fromarray(array).save(path)


def safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    return safe.strip("._") or "metric"


def prefix_role_tail(value: str, prefix: str) -> str | None:
    if not prefix:
        return None
    if not prefix[-1].isalnum():
        if value.startswith(prefix) and len(value) > len(prefix):
            return value[len(prefix) :]
        return None
    marker = f"{prefix}_"
    if value.startswith(marker) and len(value) > len(marker):
        return value[len(marker) :]
    return None


def normalize_params(params: Any) -> Any:
    if isinstance(params, Namespace):
        return vars(params)
    if isinstance(params, dict):
        return {str(key): normalize_params(value) for key, value in params.items()}
    if isinstance(params, (list, tuple)):
        return [normalize_params(value) for value in params]
    return params
