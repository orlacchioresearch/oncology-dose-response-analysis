"""Tests for the Bliss-independence synergy calculation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.bliss_synergy_demo import calculate_bliss_synergy

ROOT = Path(__file__).resolve().parents[1]


def _additive_matrix(baseline=0.0):
    """Build a matrix that is exactly Bliss-additive (zero true synergy).

    Single-agent effects (above baseline): A -> {0.3, 0.5}, B -> {0.4, 0.6}.
    Combination effects follow E_A + E_B - E_A*E_B, then the baseline is added
    back to every well so the input mimics a non-zero assay background.
    """
    ea = {0.0: 0.0, 1.0: 0.3, 2.0: 0.5}
    eb = {0.0: 0.0, 1.0: 0.4, 2.0: 0.6}
    rows = []
    for a, va in ea.items():
        for b, vb in eb.items():
            eff = va + vb - va * vb
            obs = baseline + (1.0 - baseline) * eff  # invert the normalization
            rows.append((a, b, obs))
    return pd.DataFrame(rows, columns=["drug_a_uM", "drug_b_uM", "observed_inhibition"])


def test_additive_data_has_zero_excess():
    merged, overall = calculate_bliss_synergy(_additive_matrix(baseline=0.0))
    assert overall == pytest.approx(0.0, abs=1e-9)
    assert merged["bliss_excess"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_baseline_is_corrected():
    """A non-zero baseline must not create phantom synergy/antagonism."""
    merged, overall = calculate_bliss_synergy(_additive_matrix(baseline=0.2))
    assert overall == pytest.approx(0.0, abs=1e-9)


def test_monotherapy_and_control_wells_have_zero_excess_on_real_data():
    df = pd.read_csv(ROOT / "data" / "synthetic_synergy_matrix.csv")
    merged, _ = calculate_bliss_synergy(df)
    edge = merged[(merged.drug_a_uM == 0) | (merged.drug_b_uM == 0)]
    assert edge["bliss_excess"].abs().max() < 1e-9  # single-agent/control => exactly 0


def test_real_data_overall_score_is_mild_synergy():
    df = pd.read_csv(ROOT / "data" / "synthetic_synergy_matrix.csv")
    _, overall = calculate_bliss_synergy(df)
    assert overall > 0  # baseline-corrected score flips from the buggy negative value


def test_observed_above_expected_is_synergy():
    m = _additive_matrix(baseline=0.0)
    m.loc[(m.drug_a_uM == 2.0) & (m.drug_b_uM == 2.0), "observed_inhibition"] = 0.95  # > expected 0.80
    merged, overall = calculate_bliss_synergy(m)
    cell = merged[(merged.drug_a_uM == 2.0) & (merged.drug_b_uM == 2.0)]
    assert float(cell["bliss_excess"].iloc[0]) > 0
    assert overall > 0


def test_missing_column_raises():
    with pytest.raises(ValueError):
        calculate_bliss_synergy(pd.DataFrame({"drug_a_uM": [0], "observed_inhibition": [0.1]}))


def test_missing_control_well_raises():
    df = pd.read_csv(ROOT / "data" / "synthetic_synergy_matrix.csv")
    df = df[~((df.drug_a_uM == 0) & (df.drug_b_uM == 0))]  # drop the (0,0) control
    with pytest.raises(ValueError):
        calculate_bliss_synergy(df)
