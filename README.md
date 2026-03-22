# CausalImpact

A Python implementation of Google's [CausalImpact](https://google.github.io/CausalImpact/) (R package) for causal inference using Bayesian structural time series models.

The Gibbs sampler is written in Rust (via PyO3) for performance, while maintaining numerical compatibility (within ±5%) with the original R package.

## Features

- Bayesian structural time series (BSTS) model with local level and regression components
- Rust-native Gibbs sampler with Kalman filter and simulation smoother
- Point and cumulative effect estimation with credible intervals
- Posterior tail-area p-values
- Summary reports (tabular and narrative)
- Visualization with matplotlib (original, pointwise, cumulative panels)

## Installation

Requires Python 3.12+ and a Rust toolchain.

```bash
# Clone the repository
git clone https://github.com/YuminosukeSato/CausalImpact.git
cd CausalImpact

# Install with uv (recommended)
uv sync --all-extras

# Or install with pip (builds Rust extension via maturin)
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from causal_impact import CausalImpact

# Prepare your data: first column = response, remaining columns = covariates
data = pd.read_csv("your_data.csv", index_col="date", parse_dates=True)

# Define pre- and post-intervention periods
pre_period = ["2020-01-01", "2020-03-14"]
post_period = ["2020-03-15", "2020-04-14"]

# Run the analysis
ci = CausalImpact(data, pre_period, post_period)

# Print a summary table
print(ci.summary())

# Print a narrative report
print(ci.report())

# Plot the results
fig = ci.plot()
fig.savefig("causal_impact.png")
```

## API

### `CausalImpact(data, pre_period, post_period, model_args=None, alpha=0.05)`

| Parameter | Type | Description |
|---|---|---|
| `data` | `DataFrame` or `ndarray` | First column is the response variable, remaining columns are covariates |
| `pre_period` | `list[str \| int]` | `[start, end]` of the pre-intervention period |
| `post_period` | `list[str \| int]` | `[start, end]` of the post-intervention period |
| `model_args` | `dict` (optional) | MCMC parameters (see below) |
| `alpha` | `float` | Significance level for credible intervals (default: 0.05) |

#### Model Arguments

| Key | Default | Description |
|---|---|---|
| `niter` | 1000 | Total MCMC iterations |
| `nwarmup` | 500 | Burn-in iterations to discard |
| `nchains` | 1 | Number of MCMC chains |
| `seed` | 0 | Random seed for reproducibility |
| `prior_level_sd` | 0.01 | Prior standard deviation for the local level |
| `standardize_data` | `True` | Standardize data before fitting |

#### Methods and Properties

| Name | Returns | Description |
|---|---|---|
| `summary(output="summary")` | `str` | Tabular summary of causal effects |
| `report()` | `str` | Narrative interpretation of results |
| `plot(metrics=None)` | `Figure` | Matplotlib figure with original/pointwise/cumulative panels |
| `inferences` | `DataFrame` | Per-timestep effects, predictions, and credible intervals |
| `summary_stats` | `dict` | Aggregate statistics (effect mean, CI, p-value, etc.) |

## Architecture

```
python/causal_impact/
    __init__.py          # Public API: CausalImpact, __version__
    data.py              # DataProcessor: validation, standardization, period parsing
    main.py              # CausalImpact facade class
    analysis.py          # CausalAnalysis: effect computation, CI, p-values
    summary.py           # SummaryFormatter: tabular and narrative reports
    plot.py              # Plotter: matplotlib visualization

src/ (Rust)
    lib.rs               # PyO3 entry point: run_gibbs_sampler()
    sampler.rs            # Gibbs sampler (R bsts-compatible algorithm)
    kalman.rs             # Kalman filter and simulation smoother
    state_space.rs        # State space model representation
    distributions.rs      # Posterior sampling distributions
```

## Running Tests

```bash
# Python tests (56 tests)
uv run pytest tests/ -v

# Rust tests (13 tests)
cargo test
```

## License

MIT
