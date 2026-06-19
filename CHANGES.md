# Changes — review findings addressed

Every finding from the code review is resolved below. Verified by running both
scripts and the test suite (12/12 passing).

| # | Severity | Finding | Fix |
| --- | --- | --- | --- |
| 1 | High | Bliss synergy ignored the non-zero (0,0) baseline; single-agent terms included it, biasing combos toward antagonism | Effects are normalized to the untreated control before Bliss (`effect = (obs − baseline)/(1 − baseline)`). Single-agent/control wells now score exactly 0; overall score flips from **−0.016 (antagonism)** to **+0.009 (synergy)**. |
| 2 | Medium | Heatmap/results included monotherapy + (0,0) wells, showing artifact "antagonism" along both edges | Heatmap now plots **combination wells only**, with a symmetric color scale centered at 0 and per-cell value labels. |
| 3 | Medium | IC50 reported with no uncertainty or goodness-of-fit (covariance discarded) | `ic50_summary.csv` now includes **IC50 standard error, 95% CI, and R²** per cell line. |
| 4 | Low | Committed `.venv` (~9,560 files) and `.DS_Store` | Removed; packaged from a clean tree. `.gitignore` keeps envs/artifacts out. |
| 5 | Low | Unpinned dependencies; no tests | Dependencies pinned; **SciPy dependency removed** (replaced by a small built-in 4PL fitter, validated against the original results); `tests/` added (12 tests). |
| 6 | Low | `print` vs `logging`; no error handling around the fit; space in a figure filename | Both scripts use `logging`; the fit is wrapped in error handling; figure names are slugified (`dose_response_MIA_PaCa-2.png`). |

## Verify

```bash
pip install -r requirements-dev.txt
python -m pytest -q                    # 12 passed
python src/dose_response_analysis.py   # IC50 + CI + R²
python src/bliss_synergy_demo.py       # mean Bliss excess +0.009
```

## Note on the fitter
The original used `scipy.optimize.curve_fit`. To keep the project dependency-light
and fully testable in any environment, fitting now uses a compact NumPy routine
(exact linear solve for the plateaus inside a grid search over IC50/Hill, with a
Gauss-Newton covariance for the confidence intervals). It reproduces the original
SciPy IC50s to within 1% (e.g., PANC-1 1.641 vs 1.636 µM).
