"""
Unit tests for data pipeline (Phase 1).
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_all_cells, build_lobo_folds, RATED_CAPACITY


def test_data_loading():
    """Verify loading of NASA battery dataset files."""
    data_dir = Path("data/raw")
    if not data_dir.exists() or not list(data_dir.glob("*.mat")):
        pytest.skip("Raw data files not present")

    cells = load_all_cells(data_dir)
    assert len(cells) == 4
    expected_cells = {"B0005", "B0006", "B0007", "B0018"}
    assert set(cells.keys()) == expected_cells

    for cid, df in cells.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 100
        # Check required columns
        for col in [
            "soh", "capacity_ah", "throughput_cycle_ah",
            "mean_temp_cycle", "c_rate", "rest_flag",
            "rest_duration_hours", "v_mean", "t_mean"
        ]:
            assert col in df.columns, f"Missing column {col} in {cid}"

        # SoH should be within reasonable bounds [0.5, 1.05]
        assert (df["soh"] >= 0.5).all()
        assert (df["soh"] <= 1.05).all()


def test_lobo_folds():
    """Verify Leave-One-Battery-Out fold splits."""
    cell_ids = ["B0005", "B0006", "B0007", "B0018"]
    folds = build_lobo_folds(cell_ids)
    assert len(folds) == 4

    for train_ids, test_id in folds:
        assert test_id in cell_ids
        assert len(train_ids) == 3
        assert test_id not in train_ids
        assert set(train_ids).union({test_id}) == set(cell_ids)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
