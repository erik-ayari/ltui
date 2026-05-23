from pathlib import Path
import tomllib

from ltui import __version__


def test_package_version_matches_project_metadata() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text())

    assert __version__ == metadata["project"]["version"]
