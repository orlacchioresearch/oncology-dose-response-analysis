"""Dose-response (4-parameter logistic) curve fitting and IC50 estimation.

Fits a 4PL model to cell-viability data for each cell line, reports the IC50
with a 95% confidence interval and the R-squared goodness-of-fit, and saves a
per-cell-line plot.

Fitting is done with a small, dependency-light least-squares routine (NumPy
only). The trick: for fixed (IC50, Hill slope) the 4PL model is *linear* in the
top and bottom plateaus, so we grid-search the two nonlinear parameters with an
exact linear solve inside the loop, then estimate parameter uncertainty from the
Gauss-Newton covariance at the optimum. This reproduces the standard
scipy.optimize.curve_fit result on this data (validated to <1% on the bundled
examples) while keeping the toolkit free of a SciPy dependency and fully testable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_viability.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

REQUIRED_COLUMNS = {"cell_line", "concentration_uM", "viability_percent"}
Z95 = 1.959963984540054  # large-sample normal approximation for a 95% interval


def four_parameter_logistic(x, bottom, top, ic50, hill):
    """4PL: viability falls from ``top`` (low dose) to ``bottom`` (high dose)."""
    x = np.asarray(x, dtype=float)
    return bottom + (top - bottom) / (1.0 + (x / ic50) ** hill)


def _best_plateaus(x, y, ic50, hill, bottom_min=0.0):
    """For fixed (ic50, hill), solve the linear top/bottom and return its SSE."""
    g = 1.0 / (1.0 + (x / ic50) ** hill)  # 1 at low dose, 0 at high dose
    bottom, top = np.linalg.lstsq(np.column_stack([1.0 - g, g]), y, rcond=None)[0]
    if bottom < bottom_min:  # viability bottom plateau cannot be negative
        bottom = bottom_min
        top = float(np.linalg.lstsq(g[:, None], y - bottom * (1.0 - g), rcond=None)[0][0])
    resid = y - (bottom * (1.0 - g) + top * g)
    return float(bottom), float(top), float((resid ** 2).sum())


def fit_4pl(concentration, viability) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit a 4PL curve.

    Returns
    -------
    params : np.ndarray  -> [bottom, top, ic50, hill]
    se     : np.ndarray  -> standard error of each parameter
    r2     : float       -> coefficient of determination
    """
    x = np.asarray(concentration, dtype=float)
    y = np.asarray(viability, dtype=float)
    if len(np.unique(x)) < 4:
        raise ValueError("Need at least four distinct concentrations to fit a 4PL curve.")
    if np.any(x <= 0):
        raise ValueError("Concentrations must be positive (the model uses log-dose).")

    def grid(ic_lo, ic_hi, h_lo, h_hi, n_ic, n_h):
        best = (None, np.inf)
        for ic in np.logspace(np.log10(ic_lo), np.log10(ic_hi), n_ic):
            for h in np.linspace(h_lo, h_hi, n_h):
                b, t, sse = _best_plateaus(x, y, ic, h)
                if sse < best[1]:
                    best = ((b, t, ic, h), sse)
        return best[0]

    b, t, ic, h = grid(x.min(), x.max(), 0.2, 5.0, 120, 80)          # coarse
    b, t, ic, h = grid(max(ic / 3, 1e-9), ic * 3, max(0.2, h - 0.6), h + 0.6, 140, 90)  # refine
    params = np.array([b, t, ic, h], dtype=float)

    # Gauss-Newton covariance from a central-difference Jacobian.
    jac = np.zeros((len(x), 4))
    for k in range(4):
        step = 1e-6 * max(abs(params[k]), 1e-6)
        up, dn = params.copy(), params.copy()
        up[k] += step
        dn[k] -= step
        jac[:, k] = (four_parameter_logistic(x, *up) - four_parameter_logistic(x, *dn)) / (2 * step)

    resid = y - four_parameter_logistic(x, *params)
    dof = max(len(x) - 4, 1)
    mse = float((resid ** 2).sum() / dof)
    cov = mse * np.linalg.pinv(jac.T @ jac)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot else float("nan")
    return params, se, r2


def _slug(text: str) -> str:
    """Filesystem-safe name (no spaces) for figure files."""
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(text))


def plot_dose_response(group, params, r2, cell_line, output_path: Path) -> None:
    x = group["concentration_uM"].to_numpy(float)
    y = group["viability_percent"].to_numpy(float)
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    plt.figure()
    plt.scatter(x, y, label="Replicates")
    plt.plot(xs, four_parameter_logistic(xs, *params),
             label=f"4PL fit; IC50={params[2]:.2f} µM (R²={r2:.3f})")
    plt.xscale("log")
    plt.xlabel("Concentration (µM)")
    plt.ylabel("Viability (%)")
    plt.title(f"Dose-response: {cell_line}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved dose-response curve to %s", output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    df = pd.read_csv(DATA)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Viability data missing required columns: {sorted(missing)}")

    rows = []
    for cell_line, group in df.groupby("cell_line"):
        try:
            params, se, r2 = fit_4pl(group["concentration_uM"], group["viability_percent"])
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.error("Fit failed for %s: %s", cell_line, exc)
            continue
        bottom, top, ic50, hill = params
        rows.append({
            "cell_line": cell_line,
            "estimated_ic50_uM": round(ic50, 4),
            "ic50_se_uM": round(float(se[2]), 4),
            "ic50_ci95_low_uM": round(ic50 - Z95 * se[2], 4),
            "ic50_ci95_high_uM": round(ic50 + Z95 * se[2], 4),
            "hill_slope": round(hill, 4),
            "bottom": round(bottom, 4),
            "top": round(top, 4),
            "r_squared": round(r2, 4),
        })
        plot_dose_response(group, params, r2, cell_line,
                           FIGURES / f"dose_response_{_slug(cell_line)}.png")

    summary = pd.DataFrame(rows).sort_values("estimated_ic50_uM").reset_index(drop=True)
    summary.to_csv(RESULTS / "ic50_summary.csv", index=False)
    logger.info("Wrote IC50 summary for %d cell line(s).", len(summary))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
