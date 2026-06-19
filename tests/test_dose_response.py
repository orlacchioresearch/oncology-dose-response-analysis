"""Tests for the 4PL dose-response fitting."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dose_response_analysis import (
    _slug,
    fit_4pl,
    four_parameter_logistic,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fit_recovers_known_parameters():
    x = np.logspace(-3, 1.5, 10)  # 0.001 .. ~30 uM
    true = dict(bottom=0.0, top=100.0, ic50=2.0, hill=1.2)
    y = four_parameter_logistic(x, **true)
    params, se, r2 = fit_4pl(x, y)
    bottom, top, ic50, hill = params
    assert ic50 == pytest.approx(2.0, rel=0.05)
    assert hill == pytest.approx(1.2, rel=0.10)
    assert r2 > 0.999


def test_fit_on_real_data_is_good():
    df = pd.read_csv(ROOT / "data" / "synthetic_viability.csv")
    for _, g in df.groupby("cell_line"):
        params, se, r2 = fit_4pl(g.concentration_uM, g.viability_percent)
        ic50 = params[2]
        assert r2 > 0.95
        assert g.concentration_uM.min() <= ic50 <= g.concentration_uM.max()  # not extrapolated
        assert se[2] > 0  # a usable standard error was produced


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        fit_4pl([0.1, 1.0, 10.0], [100, 50, 0])  # only 3 distinct concentrations


def test_nonpositive_concentration_raises():
    with pytest.raises(ValueError):
        fit_4pl([0.0, 0.1, 1.0, 10.0], [100, 90, 50, 0])


def test_slug_removes_spaces():
    assert _slug("MIA PaCa-2") == "MIA_PaCa-2"
    assert " " not in _slug("Cell Line 1")
