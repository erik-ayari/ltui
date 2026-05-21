# ltui

Terminal UI for live and posthoc visualization of PyTorch Lightning `CSVLogger` metrics.

`ltui` is built for SSH and tmux workflows: no browser, no server process, no TensorBoard dependency. Point it at a log root and it recursively finds Lightning `metrics.csv` files, lets you switch runs and metrics from the keyboard, and renders a single focused terminal plot.

## Features

- Recursively discovers Lightning `metrics.csv` files in nested experiment folders
- Live refresh for growing CSV files, with pause and manual rescan
- Compare multiple runs in one plot
- Groups `train_` and `val_` metric families by default
- Step/epoch x-axis switching with integer ticks
- Optional EMA smoothing and log scaling
- Fuzzy selectors for runs, metrics, and YAML configs
- Per-root UI state under `~/.local/state/ltui/`

## Installation

After the first PyPI release:

```bash
pip install ltui
```

### From source

Create a dedicated micromamba environment and install the package editable:

```bash
git clone https://github.com/erik-ayari/ltui.git
cd ltui

micromamba create -n ltui python=3.11 -y
micromamba activate ltui
pip install -e ".[dev]"
```

This keeps runtime and development dependencies inside the `ltui` environment.

### From any project directory

Activate the environment, then run `ltui` against the log root you want to inspect:

```bash
micromamba activate ltui
ltui /path/to/log/root
```

Optional shell helper:

```bash
ltui() {
  micromamba run -n ltui ltui "$@"
}
```

With that function in your shell config, `ltui /path/to/log/root` works from any directory without manually activating the environment.

## Usage

```bash
ltui ./lightning_logs
ltui /data/experiments/my_model
ltui ~/runs/my_experiment
```

On startup, `ltui` selects the latest modified run and chooses a default loss metric when one is available.

## Supported Log Layouts

`ltui` targets PyTorch Lightning `CSVLogger` output. Every discovered `metrics.csv` is treated as one selectable run/version.

Supported examples:

```text
lightning_logs/version_0/metrics.csv
run_a/version_0/metrics.csv
run_a/lightning_logs/version_0/metrics.csv
experiments/group_1/run_a/lightning_logs/version_3/metrics.csv
```

Display names come from paths relative to the scanned root.

## Keybindings

| Key | Action |
| --- | --- |
| `m` | Open metric selector |
| `r` | Open run/version selector |
| `c` | Open config viewer for runs with a unique YAML config |
| `/` | Fuzzy search inside selector |
| `arrow keys` | Navigate selector |
| `space` | Toggle selection in selector, pause/resume on main screen |
| `enter` | Apply selector |
| `n` | Next selected metric/family |
| `p` | Previous selected metric/family |
| `a` | Toggle x-axis between step and epoch |
| `d` | Toggle dark/light plot theme |
| `s` | Toggle smoothing |
| `x` | Toggle log-x |
| `y` | Toggle log-y |
| `R` | Force rescan |
| `q` | Quit |

## Metric Handling

Metric columns are numeric columns except `step` and `epoch`. Rows with `NaN` for the selected metric are dropped, and points are sorted by x-axis before plotting.

Grouped mode is enabled by default. `train_loss_step`, `train_loss_epoch`, and `val_loss` appear as family `loss`; selecting `loss` plots whichever train/val sides exist. Use `a` to switch between step and epoch. On the step axis, validation epoch metrics use the `step` value from their CSV row, which places validation at the training step where it was logged.

## Development

```bash
micromamba activate ltui
pytest
python -m build
twine check dist/*
```

The CLI entrypoint is:

```bash
ltui /path/to/log/root
```

## Known Limitations

- CSV files are reread whole during refresh.
- Manual zoom and pan are not implemented.
- Only PyTorch Lightning `CSVLogger` style `metrics.csv` files are targeted.
- TensorBoard, WandB, Aim, browser UI, and server mode are out of scope for v1.
