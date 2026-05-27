# ltui

[![PyPI](https://img.shields.io/pypi/v/ltui.svg)](https://pypi.org/project/ltui/)

`ltui` is a terminal-native training monitor and lightweight PyTorch Lightning logger. It gives you TensorBoard-like live plots, metric inspection, and run comparison directly in the terminal, which makes it especially useful for headless training on remote machines over SSH or inside tmux.

<p>
  <a href="https://github.com/erik-ayari/ltui/blob/main/docs/screenshots/multiplot.png">
    <img src="https://raw.githubusercontent.com/erik-ayari/ltui/main/docs/screenshots/multiplot.png" alt="Grouped multiplot view in ltui" width="100%">
  </a>
</p>
<p>
  <a href="https://github.com/erik-ayari/ltui/blob/main/docs/screenshots/loss-log-scale-comparison.png">
    <img src="https://raw.githubusercontent.com/erik-ayari/ltui/main/docs/screenshots/loss-log-scale-comparison.png" alt="Run comparison plot in ltui" width="100%">
  </a>
</p>
<p>
  <a href="https://github.com/erik-ayari/ltui/blob/main/docs/screenshots/metric-selector.png">
    <img src="https://raw.githubusercontent.com/erik-ayari/ltui/main/docs/screenshots/metric-selector.png" alt="Hierarchical metric selector in ltui" width="100%">
  </a>
</p>
<p>
  <a href="https://github.com/erik-ayari/ltui/blob/main/docs/screenshots/config-viewer.png">
    <img src="https://raw.githubusercontent.com/erik-ayari/ltui/main/docs/screenshots/config-viewer.png" alt="YAML config viewer in ltui" width="100%">
  </a>
</p>

## Highlights

- Live terminal monitoring for PyTorch Lightning runs without a browser or server process
- Optional `LtuiLogger` with deterministic metric hierarchy, train/validation roles, image logging, and plain file output
- Compatibility with existing Lightning `CSVLogger` `metrics.csv` directories
- Automatic train/validation grouping in one plot
- Grouped multiplot pages that keep sibling metrics together under shared parent titles
- Multi-run comparison with color-coded run legends
- Step and epoch x-axis modes with Lightning-friendly alignment
- Log scaling and EMA smoothing for noisy or wide-range metrics
- YAML config inspection for runs with associated model or training configs
- Keyboard-first selectors with fuzzy search

## Installation

Install the TUI:

```bash
pip install ltui
```

Install the optional Lightning logger dependencies in your training environment:

```bash
pip install "ltui[logger]"
```

The TUI itself does not require PyTorch Lightning. Image viewing uses `feh`, so install it separately if you want to browse logged image streams from inside `ltui`.

## Usage

Point `ltui` at any directory containing Lightning logs:

```bash
ltui /path/to/log/root
```

Examples:

```bash
ltui ./outputs
ltui ./lightning_logs
ltui /data/experiments/stage1
```

On startup, `ltui` recursively discovers supported run directories, selects the latest modified run by default, and chooses a loss metric when one is available. The command can be run from any project directory as long as the log root path is reachable.

The preferred x-axis is `step`, then `epoch`, then row index. Press `a` to toggle between step and epoch mode. For Lightning validation metrics logged at epoch boundaries, `ltui` uses the row's step value in step mode so validation points align with the training step where validation was logged.

## Lightning Logger

`LtuiLogger` is the recommended logger for new runs. It writes a small manifest, one narrow CSV file per metric series, optional image folders, and hyperparameters as JSON. The format stays easy to copy from remote machines and inspect with standard tools, while giving `ltui` explicit structure for grouping and display.

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

The first path node is the role. By default, `train` and `val` are recognized as train/validation prefixes. The last node is the plotted metric name. Intermediate nodes define the hierarchy used by the metric selector and grouped multiplot layout.

For the example above, the metric selector shows:

```text
loss
  kl
  recon

density
  mu
```

The multiplot view uses the same structure: sibling metrics such as `loss/kl` and `loss/recon` are displayed together under a shared `loss` title, while each subplot only uses the short leaf title.

Nodes can be metrics and groups at the same time:

```python
self.log("train/pose_head/loss", pose_loss)
self.log("train/pose_head/loss/orientation_cosine_error", orientation_error)
```

The selector displays this as:

```text
pose_head
  loss  [metric]
    orientation_cosine_error
```

Prefixes and separators are configurable:

```python
logger = LtuiLogger(
    save_dir="outputs",
    name="stage1",
    train_prefix="fit",
    val_prefix="valid",
    separator="/",
)

self.log("fit/loss/kl", train_kl)
self.log("valid/loss/kl", val_kl)
```

The logger also accepts `train_` and `val_` style names with the default prefixes, so simple metric names like `train_loss` and `val_loss` remain grouped as `loss`.

## Logger Output

A logger run looks like this:

```text
outputs/stage1/version_0/
  ltui_manifest.json
  hparams.json
  series/
    train/loss/kl.csv
    val/loss/kl.csv
    train/loss/recon.csv
    val/loss/recon.csv
```

Each metric CSV has a narrow schema:

```csv
step,epoch,wall_time,value
0,0,1780000000.1,1.23
100,0,1780000002.4,0.97
```

This structure lets `ltui` reread only the files it needs for the selected metrics and leaves a clear path for more incremental loading in future versions.

## Image Logging

`LtuiLogger` can store image streams with the same path-style hierarchy:

```python
logger.log_image("train/recon/sample", image, step=global_step, epoch=current_epoch)
logger.log_image("val/recon/sample", image, step=global_step, epoch=current_epoch)
```

TensorBoard-style calls through the experiment object are also supported:

```python
logger.experiment.add_image("train/recon/sample", image_tensor, global_step=global_step)
```

Images are written under the run directory with step and epoch in the filename, so alphabetical order follows training time:

```text
outputs/stage1/version_0/
  images/
    train/recon/sample/
      step_100_epoch_0.png
      step_200_epoch_0.png
```

Press `i` in the TUI to choose an image stream. If multiple selected runs contain that stream, `ltui` asks which run/version to open, then launches `feh` on the image folder.

## CSVLogger Compatibility

Existing Lightning `CSVLogger` runs work without changing your training code. Every discovered `metrics.csv` is treated as one selectable run/version.

Supported layouts include:

```text
lightning_logs/version_0/metrics.csv
run_a/version_0/metrics.csv
run_a/lightning_logs/version_0/metrics.csv
experiments/group_1/run_a/lightning_logs/version_3/metrics.csv
```

For native `metrics.csv` files, train/validation grouping uses the standard prefixes:

```text
train_
val_
```

Examples:

```text
train_loss + val_loss -> loss
train_recon_loss + val_recon_loss -> recon_loss
train_kl + val_kl -> kl
```

Lightning step/epoch suffixes are handled as part of the same family, so `train_loss_step`, `train_loss_epoch`, and `val_loss` appear as `loss`.

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
