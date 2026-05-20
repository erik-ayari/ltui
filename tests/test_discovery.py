from pathlib import Path

from lightning_tui.discovery import discover_runs


def write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("step,train_loss\n0,1.0\n")


def test_discovers_simple_lightning_logs(tmp_path: Path) -> None:
    write_metrics(tmp_path / "lightning_logs" / "version_0" / "metrics.csv")

    runs = discover_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].display_name == "lightning_logs/version_0"
    assert runs[0].version_name == "version_0"


def test_discovers_nested_run_version(tmp_path: Path) -> None:
    write_metrics(tmp_path / "run_a" / "version_0" / "metrics.csv")

    runs = discover_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].display_name == "run_a/version_0"
    assert runs[0].parent_name == "run_a"


def test_names_nested_lightning_logs_readably(tmp_path: Path) -> None:
    write_metrics(tmp_path / "run_a" / "lightning_logs" / "version_0" / "metrics.csv")

    runs = discover_runs(tmp_path)

    assert runs[0].display_name == "run_a/version_0"
    assert runs[0].parent_name == "run_a"
