# Oncology Dose-Response & Synergy Analysis

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![Deps](https://img.shields.io/badge/deps-numpy%20%7C%20pandas%20%7C%20matplotlib-informational)](#)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](#)

A small, reproducible workflow for analyzing oncology cell-viability data, using
synthetic examples. It estimates drug potency from dose-response curves and
evaluates two-drug combination activity with Bliss independence.

> **Scope.** All data is synthetic and clearly labelled as such. This is a
> portfolio / demonstration project, **not** a validated clinical or regulatory
> analysis package.

## What it does

**Dose-response (`src/dose_response_analysis.py`)**
Fits a four-parameter logistic (4PL) curve per cell line and reports the **IC50
with a 95% confidence interval** and the **R² goodness-of-fit**, plus a plot per
cell line. The fit is dependency-light (NumPy only): for a fixed (IC50, Hill)
the 4PL is linear in its plateaus, so the two nonlinear parameters are
grid-searched with an exact linear solve inside, and parameter uncertainty comes
from the Gauss-Newton covariance at the optimum. (Validated to reproduce
`scipy.optimize.curve_fit` to <1% on the bundled data.)

**Synergy (`src/bliss_synergy_demo.py`)**
Computes Bliss-independence expected effects and **Bliss excess** (observed −
expected; >0 synergy, <0 antagonism) across a combination matrix. Effects are
**normalized to the untreated (0,0) control** first, so the non-zero assay
baseline doesn't leak into the single-agent terms — single-agent and control
wells therefore score exactly 0, and only true combination wells contribute to
the overall score and the heatmap.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/dose_response_analysis.py     # -> results/ic50_summary.csv + figures/
python src/bliss_synergy_demo.py         # -> results/bliss_synergy_results.csv + heatmap

pip install -r requirements-dev.txt && python -m pytest -q   # run the tests
```

Everything in `results/` and `figures/` is a reproducible build artifact
(git-ignored) — a clean clone regenerates all of it from the committed data.

## Example results (bundled synthetic data)

| Cell line | IC50 (µM) | 95% CI | R² |
| :-- | --: | :-- | --: |
| MIA PaCa-2 | 0.66 | 0.55–0.76 | 0.992 |
| PANC-1 | 1.64 | 1.38–1.91 | 0.992 |
| BxPC-3 | 4.43 | 2.26–6.59 | 0.983 |

Combination: mean Bliss excess **+0.009** (mild synergy), strongest at
Drug A 0.3 µM / Drug B 1.0 µM.

## Repository structure

```text
├── README.md
├── requirements.txt / requirements-dev.txt   <- pinned dependencies
├── data/
│   ├── synthetic_viability.csv                <- dose-response replicates
│   └── synthetic_synergy_matrix.csv           <- combination matrix
├── src/
│   ├── dose_response_analysis.py              <- 4PL fit, IC50 + CI, R²
│   └── bliss_synergy_demo.py                  <- baseline-corrected Bliss synergy
├── tests/                                     <- pytest unit + integration tests
├── figures/                                   <- generated plots
└── results/                                   <- generated CSVs
```

## Notes & limitations
- The 4PL `ic50` is the curve's half-maximal-effect concentration (relative
  IC50); it equals the absolute (50%-of-control) IC50 only when the bottom
  plateau is ~0, which holds here.
- IC50 confidence intervals use a large-sample normal approximation
  (±1.96·SE); for very small n a t-multiplier is slightly wider.
- Bliss independence assumes the drugs act independently; a positive excess
  flags *potential* combination benefit, not a mechanism.
