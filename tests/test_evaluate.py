"""Tests for paired statistical evaluation and censoring treatment."""

import numpy as np
import pandas as pd

from experiments.evaluate import run_statistical_analysis


def test_rul_wilcoxon_excludes_censored_fold_for_every_comparison():
    """A B0007 RUL NaN is excluded pairwise, while SoH retains all four folds."""
    rows = []
    for model, offset in [("PI-MDCNet", 0.0), ("RandomForest", 1.0)]:
        for fold, rul in zip(["B0005", "B0006", "B0007", "B0018"], [8.0, 6.0, np.nan, 4.0]):
            rows.append({
                "model": model,
                "fold": fold,
                "soh_rmse": 0.01 + offset,
                "soh_mae": 0.01 + offset,
                "rul_rmse": rul + offset if np.isfinite(rul) else np.nan,
                "rul_mae": rul + offset if np.isfinite(rul) else np.nan,
            })

    summary = run_statistical_analysis(pd.DataFrame(rows))
    rul_row = summary[summary["Metric"] == "RUL RMSE"].iloc[0]
    soh_row = summary[summary["Metric"] == "SoH RMSE"].iloc[0]
    assert rul_row["N paired"] == 3
    assert rul_row["Excluded"] == 1
    assert soh_row["N paired"] == 4
