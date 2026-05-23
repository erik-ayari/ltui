# ltui

[![PyPI](https://img.shields.io/pypi/v/ltui.svg)](https://pypi.org/project/ltui/)

TensorBoard-like live monitoring for PyTorch Lightning `CSVLogger` metrics, directly in the terminal.

`ltui` turns Lightning `metrics.csv` files into a focused terminal plot for live monitoring, inspection, and run comparison, similar in spirit to tools like TensorBoard. Because it runs entirely in the terminal, it is especially useful for headless monitoring on remote machines over SSH, inside tmux, and without a browser or server process.

<p>
  <a href="https://github.com/erik-ayari/ltui/blob/main/docs/screenshots/multiplot.png">
    <img src="https://raw.githubusercontent.com/erik-ayari/ltui/main/docs/screenshots/multiplot.png" alt="Multiplot grid in ltui" width="100%">
  </a>
</p>
<p>
  <a href="https://github.com/erik-ayari/ltui/blob/main/docs/screenshots/loss-log-scale-comparison.png">
    <img src="https://raw.githubusercontent.com/erik-ayari/ltui/main/docs/screenshots/loss-log-scale-comparison.png" alt="Run comparison plot in ltui" width="100%">
  </a>
</p>
<p>
  <a href="https://github.com/erik-ayari/ltui/blob/main/docs/screenshots/metric-selector.png">
    <img src="https://raw.githubusercontent.com/erik-ayari/ltui/main/docs/screenshots/metric-selector.png" alt="Metric selector in ltui" width="100%">
  </a>
</p>
<p>
  <a href="https://github.com/erik-ayari/ltui/blob/main/docs/screenshots/config-viewer.png">
    <img src="https://raw.githubusercontent.com/erik-ayari/ltui/main/docs/screenshots/config-viewer.png" alt="YAML config viewer in ltui" width="100%">
  </a>
</p>

## Highlights

- Live in-terminal visualization of PyTorch Lightning training metrics
- Automatic train/validation metric grouping in one plot
- Multiplot mode for viewing several selected metrics at once
- Multi-run comparison with readable run legends
- Step and epoch x-axis modes with Lightning-friendly alignment
- Log scaling and EMA smoothing for noisy or wide-range metrics
- YAML config inspection for runs with associated model/training configs
- Keyboard-first selectors with fuzzy search

## Installation

```bash
pip install ltui
```

## Quick Start

Point `ltui` at any directory containing Lightning CSV logs:

```bash
ltui /path/to/log/root
```

Examples:

```bash
ltui ./lightning_logs
ltui /data/experiments/my_model
ltui ~/runs/my_experiment
```

On startup, `ltui` recursively discovers `metrics.csv` files, selects the latest modified run, and chooses a loss metric when one is available.

## What It Reads

`ltui` targets PyTorch Lightning `CSVLogger` output. Each discovered `metrics.csv` is treated as one selectable run/version.

Supported layouts include:

```text
lightning_logs/version_0/metrics.csv
run_a/version_0/metrics.csv
run_a/lightning_logs/version_0/metrics.csv
experiments/group_1/run_a/lightning_logs/version_3/metrics.csv
```

Display names are derived from paths relative to the scanned root.

## Metric Grouping

Train/validation grouping is enabled by default. Metrics are grouped when they use the standard prefixes:

```text
train_
val_
```

For example:

```text
train_loss + val_loss -> loss
train_recon_loss + val_recon_loss -> recon_loss
train_kl + val_kl -> kl
```

Lightning step/epoch suffixes are handled as part of the same family, so `train_loss_step`, `train_loss_epoch`, and `val_loss` appear as `loss`.

When the x-axis is `step`, validation epoch metrics use the `step` value from their CSV row. This places validation points at the training step where validation was logged.

## Controls

| Key | Action |
| --- | --- |
| `r` | Open run/version selector |
| `c` | Open config viewer for runs with a unique YAML config |
| `m` | Open metric selector |
| `/` | Fuzzy search inside selector |
| `arrow keys` | Navigate selector or selected plot in multiplot mode; left/right jump between models in run/config selectors |
| `space` | Toggle selection in selector, open multiplot on main screen |
| `enter` | Apply selector, focus selected plot in multiplot mode |
| `escape` | Clear plot selection in multiplot mode |
| `n` | Next selected metric/family |
| `p` | Previous selected metric/family |
| `a` | Toggle x-axis between step and epoch |
| `d` | Toggle dark/light plot theme |
| `s` | Toggle smoothing |
| `x` | Toggle log-x |
| `y` | Toggle log-y |
| `q` | Quit |
