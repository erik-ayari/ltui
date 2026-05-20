# lightning-tui

Terminal UI for live and posthoc PyTorch Lightning `CSVLogger` metrics. It uses Textual and plotext, so it works over SSH and tmux without a browser.

## Environment

Create and activate a dedicated micromamba environment:

```bash
micromamba create -n lightning-tui python=3.11 -y
micromamba activate lightning-tui
```

Install the package editable from this repository:

```bash
pip install -e ".[dev]"
```

This keeps all runtime and development dependencies inside the `lightning-tui` environment. Do not rely on global Python packages.

## Usage

From any project directory:

```bash
micromamba activate lightning-tui
ltui /path/to/log/root
```

Example:

```bash
ltui ./lightning_logs
ltui /data/experiments/my_model
```

Optional shell helper:

```bash
ltui() {
  micromamba run -n lightning-tui ltui "$@"
}
```

With that function in your shell config, run:

```bash
ltui /path/to/log/root
```

## Expected Log Layouts

`ltui` recursively scans for files named `metrics.csv`. Each file is one selectable run/version.

Supported examples:

```text
lightning_logs/version_0/metrics.csv
run_a/version_0/metrics.csv
run_a/lightning_logs/version_0/metrics.csv
experiments/group_1/run_a/lightning_logs/version_3/metrics.csv
```

Display names are derived from paths relative to the scanned root.

## Keybindings

```text
m       open metric selector
r       open run/version selector
/       fuzzy search inside selector
arrows  navigate selector
space   toggle selection in selector, pause/resume on main screen
enter   apply selector
n       next selected metric/family
p       previous selected metric/family
c       toggle compare mode
g       toggle grouped train/val metric-family mode
s       toggle smoothing
x       toggle log-x
y       toggle log-y
R       force rescan
q       quit
```

Grouped mode is enabled by default. `train_loss` and `val_loss` appear as family `loss`; selecting `loss` plots whichever sides are available.

## Behavior

- Preferred x-axis: `step`
- Fallback x-axis: `epoch`
- Final fallback: row index
- Metric columns: numeric columns except `step` and `epoch`
- Rows with `NaN` for the selected metric are dropped
- Points are sorted by x-axis before plotting
- Smoothing is EMA with alpha `0.2`
- Log scaling drops nonpositive points and reports the count
- State is stored per root under `~/.local/state/lightning-tui/`

## Known Limitations

- CSV files are reread whole during refresh.
- No manual zoom or pan in v1.
- Only PyTorch Lightning `CSVLogger` style `metrics.csv` files are targeted.
- No TensorBoard, WandB, Aim, browser UI, or server mode.
