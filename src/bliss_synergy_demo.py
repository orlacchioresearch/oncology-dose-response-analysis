"""Bliss-independence synergy analysis for a drug-combination matrix.

Effects are first normalized to the no-treatment (0,0) control so that the
zero-dose effect is exactly 0 -- the scale Bliss independence assumes. Without
this step a non-zero assay baseline (the (0,0) well here shows ~7.5% inhibition)
leaks into every single-agent term, which biases all combination wells toward
apparent antagonism and gives single-agent / no-drug wells a nonsensical
non-zero "synergy" value.

Bliss independence: for fractional effects E_A and E_B, the expected combined
effect is E_A + E_B - E_A * E_B. Bliss excess = observed - expected
(> 0 synergy, < 0 antagonism). The overall score averages excess over true
combination wells only (both drugs present).
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
DATA = ROOT / "data" / "synthetic_synergy_matrix.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

REQUIRED_COLUMNS = {"drug_a_uM", "drug_b_uM", "observed_inhibition"}


def calculate_bliss_synergy(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Baseline-correct, compute Bliss expected/excess, and the overall score."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Synergy data missing required columns: {sorted(missing)}")

    df = df.copy()
    base = df[(df.drug_a_uM == 0) & (df.drug_b_uM == 0)]
    if base.empty:
        raise ValueError("No untreated (0,0) control well found; cannot baseline-correct.")
    baseline = float(base["observed_inhibition"].mean())
    if baseline >= 1.0:
        raise ValueError("Baseline inhibition >= 1; cannot normalize.")

    # Normalize so the (0,0) effect maps to 0 and full inhibition stays at 1.
    df["effect"] = (df["observed_inhibition"] - baseline) / (1.0 - baseline)

    mono_a = df[df.drug_b_uM == 0][["drug_a_uM", "effect"]].rename(columns={"effect": "effect_a"})
    mono_b = df[df.drug_a_uM == 0][["drug_b_uM", "effect"]].rename(columns={"effect": "effect_b"})
    merged = df.merge(mono_a, on="drug_a_uM", how="left").merge(mono_b, on="drug_b_uM", how="left")
    if merged[["effect_a", "effect_b"]].isna().any().any():
        raise ValueError("Missing a single-agent reference for some wells (incomplete dose grid).")

    merged["bliss_expected"] = merged.effect_a + merged.effect_b - merged.effect_a * merged.effect_b
    merged["bliss_excess"] = merged.effect - merged.bliss_expected

    combo = merged[(merged.drug_a_uM > 0) & (merged.drug_b_uM > 0)]
    overall_score = float(combo["bliss_excess"].mean())
    return merged, overall_score


def plot_synergy_heatmap(merged: pd.DataFrame, overall_score: float, output_path: Path) -> None:
    """Heatmap of Bliss excess over the true combination wells only."""
    combo = merged[(merged.drug_a_uM > 0) & (merged.drug_b_uM > 0)]
    grid = combo.pivot(index="drug_b_uM", columns="drug_a_uM", values="bliss_excess").sort_index()
    vals = grid.values
    vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
    vmax = vmax or 1.0

    plt.figure(figsize=(8, 6))
    im = plt.imshow(vals, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="Bliss excess  (synergy > 0, antagonism < 0)")
    plt.xticks(range(len(grid.columns)), [f"{c:g}" for c in grid.columns])
    plt.yticks(range(len(grid.index)), [f"{r:g}" for r in grid.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                plt.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    plt.xlabel("Drug A (µM)")
    plt.ylabel("Drug B (µM)")
    plt.title(f"Bliss synergy — combination wells only\nMean Bliss excess: {overall_score:+.3f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved synergy heatmap to %s", output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    logger.info("Loading combination matrix from %s", DATA)
    df = pd.read_csv(DATA)
    results_df, overall_score = calculate_bliss_synergy(df)
    logger.info("Overall Bliss synergy score (combination wells): %+.4f", overall_score)

    out_csv = RESULTS / "bliss_synergy_results.csv"
    results_df.to_csv(out_csv, index=False)
    logger.info("Saved numerical synergy results to %s", out_csv)

    plot_synergy_heatmap(results_df, overall_score, FIGURES / "bliss_synergy_heatmap.png")

    combo = results_df[(results_df.drug_a_uM > 0) & (results_df.drug_b_uM > 0)]
    logger.info("Top synergistic combinations:")
    print(combo.sort_values("bliss_excess", ascending=False)
          .head(5)[["drug_a_uM", "drug_b_uM", "bliss_excess"]].to_string(index=False))


if __name__ == "__main__":
    main()
