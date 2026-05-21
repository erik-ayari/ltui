from __future__ import annotations

import argparse
from pathlib import Path

from .app import LightningTuiApp
from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ltui",
        description="Terminal UI for PyTorch Lightning CSVLogger metrics.",
    )
    parser.add_argument("root", type=Path, help="Log root to scan recursively.")
    parser.add_argument("--version", action="version", version=f"ltui {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    LightningTuiApp(args.root).run()
