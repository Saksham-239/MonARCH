"""
Unit tests for data pipeline and causal SLAC decomposition (Phase 1 & Phase 2).
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_all_cells,
    build_lobo_folds,
    causal_slac_decomposition,
    get_adapter,
    ADAPTER_REGISTRY,
    RATED_CAPACITY,
)


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
            "rest_duration_hours", "v_mean", "t_mean",
            "slac_trend", "slac_regen",
        ]:
            assert col in df.columns, f"Missing column {col} in {cid}"

        # SoH should be within reasonable bounds [0.5, 1.05]
        assert (df["soh"] >= 0.5).all()
        assert (df["soh"] <= 1.05).all()


def test_causal_slac_decomposition_invariance():
    """Verify strictly causal SLAC decomposition invariance.

    Proves that for any cycle t and any future horizon k > 0:
        SLAC(capacity[:t])[-1] == SLAC(capacity[:t+k])[t-1]
    Zero future information leaks into cycle t.
    """
    np.random.seed(42)
    n = 120
    # Simulated capacity with degradation and rest jumps
    base_degradation = 2.0 - np.linspace(0, 0.6, n)
    noise = np.random.normal(0, 0.005, n)
    capacity = base_degradation + noise

    rest_flags = np.zeros(n, dtype=int)
    # Rest events at cycles 25, 60, 90
    rest_flags[[25, 60, 90]] = 1
    capacity[25:] += 0.035
    capacity[60:] += 0.025
    capacity[90:] += 0.020

    # Test across multiple cycle indices t and multiple future horizons k
    for t in [10, 25, 30, 60, 75, 90, 100]:
        for k in [1, 5, 10, 20]:
            if t + k > n:
                continue
            trend_short, regen_short = causal_slac_decomposition(capacity[:t], rest_flags[:t])
            trend_long, regen_long = causal_slac_decomposition(capacity[:t + k], rest_flags[:t + k])

            # Invariant check: index t-1 must be identical
            np.testing.assert_allclose(
                trend_short[-1],
                trend_long[t - 1],
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Causal trend violation at t={t}, k={k}",
            )
            np.testing.assert_allclose(
                regen_short[-1],
                regen_long[t - 1],
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Causal regeneration violation at t={t}, k={k}",
            )


def test_multi_dataset_adapters():
    """Verify that NASA, CALCE, and Oxford adapters exist and produce valid schemas."""
    data_dir = Path("data/raw")
    for name in ["nasa", "calce", "oxford"]:
        adapter = get_adapter(name)
        assert adapter.dataset_name == name
        assert adapter.rated_capacity_ah > 0.0
        cell_ids = adapter.get_cell_ids()
        assert len(cell_ids) >= 4

        # Test single cell loading
        test_cell = cell_ids[0]
        df = adapter.load_cell(test_cell, data_dir)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in ["soh", "capacity_ah", "slac_trend", "slac_regen", "rest_flag"]:
            assert col in df.columns, f"Adapter {name} missing column {col}"


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
