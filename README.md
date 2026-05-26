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
- Structured image logging with `feh` browsing for generated samples or reconstructions
- YAML config inspection for runs with associated model/training configs
- Keyboard-first selectors with fuzzy search

## Installation

```bash
pip install ltui
```

The TUI itself does not require PyTorch Lightning. To install the optional Lightning logger dependency in a training environment:

```bash
pip install "ltui[logger]"
```

## Quick Start

Point `ltui` at any directory containing Lightning logs:

```bash
ltui /path/to/log/root
```

Examples:

```bash
ltui ./lightning_logs
ltui /data/experiments/my_model
ltui ~/runs/my_experiment
```

On startup, `ltui` recursively discovers native Lightning `metrics.csv` files and ltui logger manifests, selects the latest modified run, and chooses a loss metric when one is available.

## What It Reads

`ltui` reads two simple file-based formats:

- PyTorch Lightning `CSVLogger` output, where each discovered `metrics.csv` is one selectable run/version.
- The optional `LtuiLogger` format, where each discovered `ltui_manifest.json` describes one run/version with one small CSV file per metric series.

Supported layouts include:

```text
lightning_logs/version_0/metrics.csv
run_a/version_0/metrics.csv
run_a/lightning_logs/version_0/metrics.csv
experiments/group_1/run_a/lightning_logs/version_3/metrics.csv
ltui_logs/version_0/ltui_manifest.json
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

## LtuiLogger

`LtuiLogger` is an optional PyTorch Lightning logger that writes a manifest plus per-metric CSV files and image folders. The format is still plain files, so it is easy to copy from remote machines, inspect with shell tools, and load with pandas.

```python
from lightning.pytorch import Trainer
from ltui.lightning import LtuiLogger

logger = LtuiLogger(
    save_dir="outputs",
    name="stage1",
)

trainer = Trainer(logger=logger)
```

Log metrics with path-style names:

```python
self.log("train/loss/kl", train_kl)
self.log("val/loss/kl", val_kl)
self.log("train/loss/recon", train_recon)
self.log("val/loss/recon", val_recon)
self.log("train/density/mu", train_mu)
self.log("val/density/mu", val_mu)
```

The first path node is the train/validation role. The last node is the plotted metric. Intermediate nodes define the metric hierarchy shown in the `m` selector. For the example above, the selector shows `loss` with `kl` and `recon`, and `density` with `mu`.

Nodes can also be metrics and groups at the same time:

```python
self.log("train/pose_head/loss", pose_loss)
self.log("train/pose_head/loss/orientation_cosine_error", orientation_error)
```

The metric selector displays this as:

```text
pose_head
  loss  [metric]
    orientation_cosine_error
```

Prefixes and separators are configurable:

```python
logger = LtuiLogger(
    save_dir="outputs",
    train_prefix="fit",
    val_prefix="valid",
    separator="/",
)

self.log("fit/loss/kl", train_kl)
self.log("valid/loss/kl", val_kl)
```

The logger writes:

```text
stage1/version_0/
  ltui_manifest.json
  series/
    train/loss/kl.csv
    val/loss/kl.csv
```

Each series CSV has a narrow schema:

```csv
step,epoch,wall_time,value
0,0,1780000000.1,1.23
100,0,1780000002.4,0.97
```

Images can use the same path-style naming:

```python
logger.log_image("train/recon/sample", image, step=global_step, epoch=current_epoch)
logger.log_image("val/recon/sample", image, step=global_step, epoch=current_epoch)
```

The logger also supports TensorBoard-style image calls through the experiment object:

```python
logger.experiment.add_image("train/recon/sample", image_tensor, global_step=global_step)
```

Images are written under the run directory with filenames prefixed by zero-padded step values, so alphabetical order follows training time:

```text
stage1/version_0/
  images/
    train/recon/sample/
      step_100_epoch_0.png
      step_200_epoch_0.png
```

Press `i` in the TUI to choose an image stream. If multiple selected runs contain that image stream, `ltui` asks which run/version to open. Image viewing launches `feh` on Linux systems where `feh` is installed.

## Controls

| Key | Action |
| --- | --- |
| `r` | Open run/version selector |
| `c` | Open config viewer for runs with a unique YAML config |
| `m` | Open metric selector |
| `i` | Open image selector and launch `feh` for the selected image stream |
| `/` | Fuzzy search inside selector |
| `arrow keys` | Navigate selector or selected plot in multiplot mode; left/right jump between models in run/config selectors |
| `space` | Toggle selection in selector, open multiplot on main screen |
| `enter` | Apply selector, focus selected plot in multiplot mode |
| `escape` | Clear plot selection in multiplot mode |
| `n` | Next selected metric/family, or next page in multiplot mode |
| `p` | Previous selected metric/family, or previous page in multiplot mode |
| `a` | Toggle x-axis between step and epoch |
| `d` | Toggle dark/light plot theme |
| `s` | Toggle smoothing |
| `x` | Toggle log-x |
| `y` | Toggle log-y |
| `q` | Quit |
