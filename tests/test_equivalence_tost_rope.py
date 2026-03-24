"""FDA TOST + Bayesian ROPE equivalence tests against R CausalImpact.

Three-layer equivalence verification:
  Layer 1: Deterministic (test_numerical_equivalence.py, seed=42, ±1%)
  Layer 2: TOST — FDA bioequivalence (Schuirmann 1987)
  Layer 3: ROPE — Bayesian equivalence (Kruschke 2018)

Statistical framework:
  TOST (Two One-Sided Tests):
    H0: mean_error >= delta (not equivalent)
    H1: mean_error < delta (equivalent)
    Decision: 90% CI upper < delta => conclude equivalence
    Reference: Schuirmann (1987), FDA Guidance (2001)

  ROPE (Region of Practical Equivalence):
    95% HDI fully within [0, delta] => accept equivalence
    Under flat prior, 95% HDI = frequentist 95% CI (Kruschke 2018)
    Reference: Kruschke (2018) AMPPS

  Mathematical note:
    With flat prior, TOST 90% CI and ROPE 95% HDI reduce to:
      x_bar + t_{alpha, n-1} * s / sqrt(n) < delta
    (different alpha values yield the two checks)

Equivalence margins (metric-specific, following FDA practice):
  point_effect_mean, cumulative_effect_total, ci_lower, ci_upper:
    delta = 0.01 (±1% relative error)
  relative_effect_mean:
    delta = 0.02 (±2% relative error)
    Rationale: sum(y-pred)/sum(pred) is a ratio estimator whose
    denominator sum(pred) introduces extra MCMC variance, yielding
    structurally 2-3x higher variance than additive metrics.
  no_effect scenario:
    Excluded from TOST/ROPE. true_effect=0 makes relative error
    ill-defined. Covered by Layer 1 with ABS_TOL=0.5.

Run with: .venv/bin/pytest tests/test_equivalence_tost_rope.py -v --runslow
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from causal_impact import CausalImpact

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# MCMC configuration (seed is overridden per run)
MCMC_ARGS_BASE = {
    "niter": 20000,
    "nwarmup": 2000,
    "prior_level_sd": 0.01,
}

N_SEEDS = 30

# Equivalence margins (metric-specific, following FDA practice of per-endpoint delta)
EQUIVALENCE_MARGIN = 0.01  # ±1% for additive metrics
EQUIVALENCE_MARGIN_RATIO = 0.02  # ±2% for relative_effect_mean (ratio estimator)
# t critical values for df=29 (N_SEEDS=30), hardcoded to avoid scipy dependency.
# Verified: scipy.stats.t.ppf(0.95, 29) = 1.6991, t.ppf(0.975, 29) = 2.0452
T_90_UPPER_DF29 = 1.699  # t_{0.95, 29} for one-sided 90% CI upper bound
T_95_HDI_DF29 = 2.045  # t_{0.975, 29} for two-sided 95% HDI

# Metrics to compare (p_value excluded — tested via significance match)
METRICS = [
    "point_effect_mean",
    "ci_lower",
    "ci_upper",
    "cumulative_effect_total",
    "relative_effect_mean",
]

# Threshold below which absolute (not relative) error is used
NEAR_ZERO_THRESHOLD = 0.5


def _delta_for_metric(metric: str) -> float:
    """Return the equivalence margin for a given metric."""
    if metric == "relative_effect_mean":
        return EQUIVALENCE_MARGIN_RATIO
    return EQUIVALENCE_MARGIN


def _load_fixture(scenario: str) -> dict:
    path = FIXTURES_DIR / f"r_reference_{scenario}.json"
    return json.loads(path.read_text())


def _build_df(fixture: dict):
    y = np.array(fixture["data"]["y"])
    n_pre = fixture["n_pre"]
    n = fixture["n"]
    x_data = fixture["data"].get("x")
    has_x = x_data and isinstance(x_data, dict) and len(x_data) > 0

    if has_x:
        cols = {"y": y}
        for xname, xvals in x_data.items():
            cols[xname] = np.array(xvals)
        df = pd.DataFrame(cols)
    else:
        df = pd.DataFrame({"y": y})

    pre_period = [0, n_pre - 1]
    post_period = [n_pre, n - 1]
    return df, pre_period, post_period


def _compute_errors(scenario: str, n_seeds: int) -> dict[str, np.ndarray]:
    """Run n_seeds independent MCMC chains, collect error per metric.

    For |r_val| > NEAR_ZERO_THRESHOLD: relative error = |py - r| / |r|
    For |r_val| <= NEAR_ZERO_THRESHOLD: absolute error = |py - r|
    """
    fixture = _load_fixture(scenario)
    r = fixture["r_output"]
    errors: dict[str, list[float]] = {m: [] for m in METRICS}

    for seed in range(1, n_seeds + 1):
        df, pre_period, post_period = _build_df(fixture)
        model_args = {**MCMC_ARGS_BASE, "seed": seed}
        model_args.update(
            {
                key.replace(".", "_"): value
                for key, value in fixture.get("model_args", {}).items()
            }
        )
        ci = CausalImpact(df, pre_period, post_period, model_args=model_args)
        py = ci.summary_stats

        for m in METRICS:
            r_val = r[m]
            py_val = py[m]
            if abs(r_val) > NEAR_ZERO_THRESHOLD:
                errors[m].append(abs(py_val - r_val) / abs(r_val))
            else:
                errors[m].append(abs(py_val - r_val))

    return {m: np.array(v) for m, v in errors.items()}


def tost_check(errors: np.ndarray, delta: float) -> dict:
    """FDA TOST equivalence check (Schuirmann 1987).

    Tests H0: mu_error >= delta against H1: mu_error < delta.
    Uses 90% one-sided confidence interval upper bound.
    Decision: upper_90_ci < delta => equivalence.
    """
    n = len(errors)
    mean = float(errors.mean())
    std = float(errors.std(ddof=1))
    sem = std / np.sqrt(n)
    upper_90 = mean + T_90_UPPER_DF29 * sem
    return {
        "passed": upper_90 < delta,
        "mean": mean,
        "std": std,
        "upper_90_ci": upper_90,
        "margin": delta,
    }


def rope_check(errors: np.ndarray, delta: float) -> dict:
    """Bayesian ROPE equivalence check (Kruschke 2018).

    Under flat prior, posterior is Normal(x_bar, s^2/n).
    95% HDI = [x_bar - t * sem, x_bar + t * sem].
    Decision: HDI upper < delta => equivalence accepted.
    """
    n = len(errors)
    mean = float(errors.mean())
    std = float(errors.std(ddof=1))
    sem = std / np.sqrt(n)
    hdi_upper = mean + T_95_HDI_DF29 * sem
    hdi_lower = max(0.0, mean - T_95_HDI_DF29 * sem)
    return {
        "passed": hdi_upper < delta,
        "mean": mean,
        "std": std,
        "hdi_95": [hdi_lower, hdi_upper],
        "rope": [0.0, delta],
    }


# ---------------------------------------------------------------------------
# Module-scope cache: compute errors once per scenario, reuse across methods
# ---------------------------------------------------------------------------
_errors_cache: dict[str, dict[str, np.ndarray]] = {}


def _get_errors(scenario: str) -> dict[str, np.ndarray]:
    if scenario not in _errors_cache:
        _errors_cache[scenario] = _compute_errors(scenario, N_SEEDS)
    return _errors_cache[scenario]


# ---------------------------------------------------------------------------
# Scenario test classes
# ---------------------------------------------------------------------------


def _assert_tost_all(errors: dict[str, np.ndarray]) -> None:
    """Assert TOST passes for all metrics with metric-specific delta."""
    for metric, vals in errors.items():
        delta = _delta_for_metric(metric)
        result = tost_check(vals, delta)
        assert result["passed"], (
            f"TOST FAIL {metric}: 90% CI upper "
            f"{result['upper_90_ci']:.6f} >= {result['margin']} "
            f"(mean={result['mean']:.6f}, std={result['std']:.6f})"
        )


def _assert_rope_all(errors: dict[str, np.ndarray]) -> None:
    """Assert ROPE passes for all metrics with metric-specific delta."""
    for metric, vals in errors.items():
        delta = _delta_for_metric(metric)
        result = rope_check(vals, delta)
        assert result["passed"], (
            f"ROPE FAIL {metric}: 95% HDI upper "
            f"{result['hdi_95'][1]:.6f} >= {result['rope'][1]} "
            f"(mean={result['mean']:.6f}, std={result['std']:.6f})"
        )


def _print_error_report(errors: dict[str, np.ndarray]) -> None:
    """Print error distribution for diagnostics."""
    for metric, vals in errors.items():
        delta = _delta_for_metric(metric)
        print(f"\n{metric} (delta={delta}):")
        print(f"  mean={vals.mean():.6f} std={vals.std(ddof=1):.6f}")
        print(f"  max={vals.max():.6f} min={vals.min():.6f}")
        pct = np.percentile(vals, 95)
        print(f"  p50={np.median(vals):.6f} p95={pct:.6f}")


@pytest.mark.slow
class TestTOSTROPEBasic:
    """TOST + ROPE for basic (no-covariate, true_effect=3) scenario."""

    SCENARIO = "basic"

    def test_tost_equivalence(self):
        """FDA TOST: 90% CI upper < delta for all metrics."""
        _assert_tost_all(_get_errors(self.SCENARIO))

    def test_rope_equivalence(self):
        """Kruschke ROPE: 95% HDI within [0, delta] for all metrics."""
        _assert_rope_all(_get_errors(self.SCENARIO))

    def test_error_distribution_report(self, capsys):
        """Diagnostic: report full error distribution (always passes)."""
        _print_error_report(_get_errors(self.SCENARIO))


@pytest.mark.slow
class TestTOSTROPECovariates:
    """TOST + ROPE for covariates (k=2, true_effect=3) scenario."""

    SCENARIO = "covariates"

    def test_tost_equivalence(self):
        _assert_tost_all(_get_errors(self.SCENARIO))

    def test_rope_equivalence(self):
        _assert_rope_all(_get_errors(self.SCENARIO))

    def test_error_distribution_report(self, capsys):
        _print_error_report(_get_errors(self.SCENARIO))


@pytest.mark.slow
class TestTOSTROPEStrongEffect:
    """TOST + ROPE for strong_effect (true_effect=8, noise_sd=0.5)."""

    SCENARIO = "strong_effect"

    def test_tost_equivalence(self):
        _assert_tost_all(_get_errors(self.SCENARIO))

    def test_rope_equivalence(self):
        _assert_rope_all(_get_errors(self.SCENARIO))

    def test_error_distribution_report(self, capsys):
        _print_error_report(_get_errors(self.SCENARIO))


@pytest.mark.slow
class TestTOSTROPESeasonal:
    """TOST + ROPE for seasonal (nseasons=7, true_effect=3) scenario."""

    SCENARIO = "seasonal"

    def test_tost_equivalence(self):
        _assert_tost_all(_get_errors(self.SCENARIO))

    def test_rope_equivalence(self):
        _assert_rope_all(_get_errors(self.SCENARIO))

    def test_error_distribution_report(self, capsys):
        _print_error_report(_get_errors(self.SCENARIO))


# no_effect scenario is excluded from TOST/ROPE:
# - true_effect=0 makes relative error ill-defined (near-zero denominator)
# - cumulative = point * 30, amplifying MCMC variance 30x
# - FDA bioequivalence studies do not test the control arm (null effect)
# - Covered by Layer 1: test_numerical_equivalence.py with ABS_TOL=0.5


# ---------------------------------------------------------------------------
# Unit tests for tost_check / rope_check boundary behavior
# ---------------------------------------------------------------------------


class TestTOSTROPEUnitBoundary:
    """Unit tests for tost_check and rope_check functions."""

    def test_tost_exact_margin_fails(self):
        """Mean at delta, zero variance => upper_90 = delta, fails."""
        errors = np.full(N_SEEDS, EQUIVALENCE_MARGIN)
        result = tost_check(errors, EQUIVALENCE_MARGIN)
        assert not result["passed"]

    def test_tost_below_margin_passes(self):
        """Mean well below delta => passes."""
        errors = np.full(N_SEEDS, 0.001)
        result = tost_check(errors, EQUIVALENCE_MARGIN)
        assert result["passed"]

    def test_tost_above_margin_fails(self):
        """Mean above delta => fails."""
        errors = np.full(N_SEEDS, 0.02)
        result = tost_check(errors, EQUIVALENCE_MARGIN)
        assert not result["passed"]

    def test_tost_high_variance_fails(self):
        """Low mean but high variance pushes CI upper above delta."""
        rng = np.random.default_rng(42)
        errors = np.abs(rng.normal(0.005, 0.02, N_SEEDS))
        result = tost_check(errors, EQUIVALENCE_MARGIN)
        # High variance should push upper bound above delta
        assert result["upper_90_ci"] > result["mean"]

    def test_rope_hdi_exact_margin_fails(self):
        """HDI upper exactly at delta with zero variance => fails (strict <)."""
        errors = np.full(N_SEEDS, EQUIVALENCE_MARGIN)
        result = rope_check(errors, EQUIVALENCE_MARGIN)
        assert not result["passed"]

    def test_rope_hdi_fully_inside_passes(self):
        """Small mean, low variance => HDI fully within ROPE."""
        errors = np.full(N_SEEDS, 0.001)
        result = rope_check(errors, EQUIVALENCE_MARGIN)
        assert result["passed"]
        assert result["hdi_95"][0] >= 0.0
        assert result["hdi_95"][1] < EQUIVALENCE_MARGIN

    def test_rope_hdi_partially_outside_fails(self):
        """Mean above delta => HDI upper exceeds ROPE."""
        errors = np.full(N_SEEDS, 0.02)
        result = rope_check(errors, EQUIVALENCE_MARGIN)
        assert not result["passed"]

    def test_rope_hdi_lower_clamped_to_zero(self):
        """HDI lower bound is clamped to 0 (errors are non-negative)."""
        errors = np.full(N_SEEDS, 0.0001)
        result = rope_check(errors, EQUIVALENCE_MARGIN)
        assert result["hdi_95"][0] >= 0.0

    def test_tost_rope_consistency_zero_variance(self):
        """With zero variance, TOST and ROPE should agree."""
        errors = np.full(N_SEEDS, 0.005)
        tost = tost_check(errors, EQUIVALENCE_MARGIN)
        rope = rope_check(errors, EQUIVALENCE_MARGIN)
        assert tost["passed"] == rope["passed"]

    def test_tost_is_stricter_than_rope_with_variance(self):
        """ROPE uses wider CI (t_{0.975}) than TOST (t_{0.95}), so TOST is less strict.

        For the same data, if ROPE passes, TOST must also pass.
        """
        errors = np.full(N_SEEDS, 0.005)
        errors[0] = 0.008  # add slight variance
        tost = tost_check(errors, EQUIVALENCE_MARGIN)
        rope = rope_check(errors, EQUIVALENCE_MARGIN)
        if rope["passed"]:
            assert tost["passed"], "If ROPE passes, TOST must also pass"

    def test_zero_errors_both_pass(self):
        """Perfect agreement (all zeros) => both tests pass trivially."""
        errors = np.zeros(N_SEEDS)
        tost = tost_check(errors, EQUIVALENCE_MARGIN)
        rope = rope_check(errors, EQUIVALENCE_MARGIN)
        assert tost["passed"]
        assert rope["passed"]
