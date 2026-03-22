"""Speed benchmark: this library vs R CausalImpact vs tfp-causalimpact.

Usage:
  python benchmarks/benchmark.py
  python benchmarks/benchmark.py > benchmarks/results.md
"""

import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from causal_impact import CausalImpact

N_REPEATS = 3


def _generate_data(
    t: int, k: int, seed: int = 42
) -> tuple[pd.DataFrame, list[int], list[int]]:
    rng = np.random.default_rng(seed)
    n_pre = int(t * 0.7)
    y = 1.0 + rng.normal(0, 1.0, t)
    y[n_pre:] += 3.0

    cols = {"y": y}
    for j in range(k):
        cols[f"x{j + 1}"] = rng.normal(0, 1, t)

    df = pd.DataFrame(cols)
    return df, [0, n_pre - 1], [n_pre, t - 1]


def benchmark_python(
    t: int, k: int, niter: int
) -> float:
    df, pre, post = _generate_data(t, k)
    start = time.perf_counter()
    CausalImpact(
        df,
        pre,
        post,
        model_args={"niter": niter, "nwarmup": niter // 2, "seed": 42},
    )
    return time.perf_counter() - start


def benchmark_r(t: int, k: int, niter: int) -> float | None:
    try:
        subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    df, pre, post = _generate_data(t, k)

    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w"
    ) as f:
        df.to_csv(f, index=False)
        csv_path = f.name

    r_script = f"""
library(CausalImpact)
df <- read.csv("{csv_path}")
t1 <- proc.time()
ci <- CausalImpact(df, c(1, {pre[1] + 1}), c({post[0] + 1}, {post[1] + 1}),
                   model.args=list(niter={niter}))
t2 <- proc.time()
cat((t2 - t1)["elapsed"])
"""
    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        Path(csv_path).unlink(missing_ok=True)
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        Path(csv_path).unlink(missing_ok=True)
        return None


def benchmark_tfp(
    t: int, k: int, niter: int
) -> float | None:
    try:
        import causalimpact as tfp_ci  # noqa: F811
    except ImportError:
        return None

    df, pre, post = _generate_data(t, k)
    dates = pd.date_range("2020-01-01", periods=t, freq="D")
    df.index = dates
    pre_str = [str(dates[pre[0]].date()), str(dates[pre[1]].date())]
    post_str = [
        str(dates[post[0]].date()),
        str(dates[post[1]].date()),
    ]

    start = time.perf_counter()
    tfp_ci.CausalImpact(df, pre_str, post_str)
    elapsed = time.perf_counter() - start
    return elapsed


def _median_time(fn, *args) -> float | None:
    times = []
    for _ in range(N_REPEATS):
        result = fn(*args)
        if result is None:
            return None
        times.append(result)
    return statistics.median(times)


def _fmt(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.3f}s"


def _speedup(base: float | None, this: float) -> str:
    if base is None:
        return "-"
    return f"{base / this:.1f}x"


def main():
    scenarios = [
        (100, 0, 1000),
        (500, 0, 1000),
        (1000, 0, 1000),
        (1000, 5, 1000),
        (5000, 0, 1000),
    ]

    print("# Benchmark Results\n")
    print(
        "| T | k | niter | "
        "This (Rust) | R (bsts) | tfp | "
        "vs R | vs tfp |"
    )
    print(
        "|--:|--:|------:|"
        "-----------:|---------:|----:|"
        "----:|-------:|"
    )

    for t, k, niter in scenarios:
        py_time = _median_time(benchmark_python, t, k, niter)
        r_time = _median_time(benchmark_r, t, k, niter)
        tfp_time = _median_time(benchmark_tfp, t, k, niter)

        print(
            f"| {t} | {k} | {niter} | "
            f"{_fmt(py_time)} | {_fmt(r_time)} | {_fmt(tfp_time)} | "
            f"{_speedup(r_time, py_time)} | "
            f"{_speedup(tfp_time, py_time)} |"
        )
        sys.stdout.flush()

    print(
        f"\nMedian of {N_REPEATS} runs. "
        "Machine: see CI environment."
    )


if __name__ == "__main__":
    main()
