"""
Unit tests for PI-MDCNet Baseline Models (Phase 3).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    BaselineModel,
    PredictionResult,
    BASELINE_REGISTRY,
    CoulombCountingModel,
    EKFModel,
    RandomForestModel,
    XGBoostModel,
    VanillaLSTMModel,
    StatisticalDecompositionModel,
    get_all_baselines,
    FEATURE_COLS,
)
from experiments.train import ELEC_COLS, STRESSOR_COLS, compute_metrics


@pytest.fixture
def dummy_data():
    """Generate synthetic battery cycle data."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "v_mean": np.random.uniform(3.0, 4.0, n),
        "v_min": np.random.uniform(2.5, 3.5, n),
        "v_max": np.random.uniform(3.5, 4.2, n),
        "v_std": np.random.uniform(0.01, 0.1, n),
        "v_slope": np.random.uniform(-0.01, 0.0, n),
        "i_mean": np.random.uniform(1.5, 2.0, n),
        "i_std": np.random.uniform(0.01, 0.1, n),
        "dq_dv_peak_v": np.random.uniform(3.2, 3.8, n),
        "dq_dv_peak_mag": np.random.uniform(0.5, 2.0, n),
        "v_ocv": np.random.uniform(3.5, 4.1, n),
        "r0": np.random.uniform(0.05, 0.15, n),
        "r1": np.random.uniform(0.3, 1.0, n),
        "c1": np.random.uniform(100, 5000, n),
        "t_mean": np.random.uniform(25, 35, n),
        "t_max": np.random.uniform(30, 45, n),
        "t_std": np.random.uniform(0.5, 3.0, n),
        "t_rise": np.random.uniform(1.0, 15.0, n),
        "throughput_cycle_ah": np.random.uniform(1.0, 2.0, n),
        "mean_temp_cycle": np.random.uniform(25, 35, n),
        "c_rate": np.random.uniform(0.5, 1.0, n),
        "rest_duration_hours": np.concatenate([np.zeros(50), np.random.uniform(2, 20, 10)]),
        "soc_at_rest_onset": np.random.uniform(0, 1, n),
        "ambient_temp_rest": np.random.uniform(22, 26, n),
        "rest_flag": np.concatenate([np.zeros(50), np.ones(10)]).astype(int),
        "soh": np.linspace(1.0, 0.7, n),
        "soc_eod": np.random.uniform(0.0, 0.1, n),
        "capacity_ah": np.linspace(2.0, 1.4, n),
        "rul": np.linspace(50, 0, n),
        "cycle_index": np.arange(n),
        "cell_id": ["TEST_CELL"] * n,
    })
    train_df = df.iloc[:40].copy()
    test_df = df.iloc[40:].copy()
    return train_df, test_df


def test_baseline_registry():
    """Verify all 6 baselines are registered."""
    baselines = get_all_baselines()
    assert len(baselines) == 6
    expected = {
        "CoulombCounting", "EKF_1RC", "RandomForest",
        "XGBoost", "VanillaLSTM", "StatDecomp"
    }
    assert set(b.name for b in baselines) == expected
    assert set(BASELINE_REGISTRY.keys()) == expected


def test_supervised_inputs_exclude_labels_and_capacity_equivalent_throughput():
    """Ensure no supervised input is a label or a re-integrated capacity proxy."""
    forbidden = {"soh", "soc_eod", "capacity_ah", "throughput_cycle_ah", "rul"}
    assert forbidden.isdisjoint(FEATURE_COLS)
    assert forbidden.isdisjoint(ELEC_COLS)
    assert forbidden.isdisjoint(STRESSOR_COLS)


def test_degenerate_target_metrics_are_unreported():
    """An all-zero SoC target is not a valid estimation benchmark."""
    metrics = compute_metrics(np.zeros(5), np.linspace(0.0, 1.0, 5), "soc_")
    assert all(np.isnan(value) for value in metrics.values())


def test_baseline_fit_predict(dummy_data):
    """Verify each baseline fits, predicts, and returns valid PredictionResult."""
    train_df, test_df = dummy_data
    n_test = len(test_df)

    for name, cls in BASELINE_REGISTRY.items():
        model = cls()
        assert isinstance(model, BaselineModel)

        model.fit(train_df)
        pred = model.predict(test_df)

        assert isinstance(pred, PredictionResult)
        assert len(pred.soh) == n_test
        assert len(pred.soc) == n_test
        assert len(pred.rul) == n_test
        assert pred.wall_clock_seconds >= 0.0

        # Model-specific validation
        if name in ("CoulombCounting", "EKF_1RC"):
            assert np.all(np.isnan(pred.soh)), f"{name} must return NaN for SoH"
            assert not np.all(np.isnan(pred.soc)), f"{name} must predict SoC"
        elif name == "VanillaLSTM":
            assert not np.all(np.isnan(pred.soh)), "LSTM must predict SoH"
            assert not np.all(np.isnan(pred.soc)), "LSTM must predict SoC"
        elif name in ("RandomForest", "XGBoost", "StatDecomp"):
            assert not np.all(np.isnan(pred.soh)), f"{name} must predict SoH"
            assert not np.all(np.isnan(pred.rul)), f"{name} must predict RUL"
            assert np.all(np.isnan(pred.soc)), f"{name} must report unimplemented SoC as NaN"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
