from pathlib import Path

from ltui.discovery import discover_runs


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


def test_discovers_ltui_manifest_run(tmp_path: Path) -> None:
    manifest = tmp_path / "run_a" / "version_0" / "ltui_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"schema_version": 1, "series": []}\n')

    runs = discover_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].display_name == "run_a/version_0"
    assert runs[0].metrics_csv_path == manifest.resolve()
    assert runs[0].logger_format == "ltui"


def test_manifest_takes_precedence_over_legacy_metrics_csv_in_same_run(tmp_path: Path) -> None:
    manifest = tmp_path / "run_a" / "version_0" / "ltui_manifest.json"
    metrics = tmp_path / "run_a" / "version_0" / "metrics.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"schema_version": 1, "series": []}\n')
    write_metrics(metrics)

    runs = discover_runs(tmp_path)

    assert [run.metrics_csv_path for run in runs] == [manifest.resolve()]


def test_discovers_unique_config_yaml_next_to_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "version_0" / "metrics.csv"
    write_metrics(metrics)
    config = tmp_path / "version_0" / "config.yaml"
    config.write_text("trainer:\n  max_epochs: 1\n")

    runs = discover_runs(tmp_path)

    assert runs[0].config_yaml_path == config.resolve()


def test_ignores_ambiguous_config_yaml_files(tmp_path: Path) -> None:
    metrics = tmp_path / "version_0" / "metrics.csv"
    write_metrics(metrics)
    (tmp_path / "version_0" / "config.yaml").write_text("a: 1\n")
    (tmp_path / "version_0" / "other.yaml").write_text("b: 2\n")

    runs = discover_runs(tmp_path)

    assert runs[0].config_yaml_path is None
