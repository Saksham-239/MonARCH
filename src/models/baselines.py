"""
PI-MDCNet v4 -- Baseline Models
================================

Seven baseline implementations for battery SoH/SoC/RUL prediction.
Each baseline exposes a common interface for fair comparison.

Baselines (spec S3):
    1. Coulomb Counting       -- open-loop integral(|I|*dt), drifts by design
    2. EKF (1RC ECM)          -- Extended Kalman Filter with state [SoC, V_p]
    3. Random Forest           -- sklearn on windowed cycle features
    4. XGBoost                 -- xgboost with Optuna hyperparameter tuning
    5. Vanilla LSTM            -- Bidirectional LSTM on multivariate windows
    6. Statistical Decomp.     -- Qin-style RTPF: trend + regen decomposition
    7. PI-MDCNet (SLAC)        -- (defined in pi_mdcnet.py, not here)

All models except LSTM and PI-MDCNet operate on tabular per-cycle features
from data_loader.py. LSTM and PI-MDCNet operate on sequences.
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# COMMON INTERFACE
# =============================================================================

@dataclass
class PredictionResult:
    """Container for model predictions.

    All arrays have shape (n_samples,). NaN indicates the model does not
    predict that quantity (e.g. Coulomb Counting only predicts SoC).
    """

    soh: NDArray[np.float64]
    soc: NDArray[np.float64]
    rul: NDArray[np.float64]
    wall_clock_seconds: float = 0.0
    """Training wall-clock time in seconds (spec correction #6)."""


class BaselineModel(ABC):
    """Abstract base class for all baseline models.

    Subclasses must implement fit() and predict().
    """

    name: str = "BaselineModel"

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, **hparams: Any) -> None:
        """Train the model on tabular per-cycle features.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training data from data_loader.process_cell(), with columns
            matching CycleFeatures fields plus 'soh', 'soc_eod', 'rul'.
        **hparams : Any
            Model-specific hyperparameters.
        """
        ...

    @abstractmethod
    def predict(self, test_data: pd.DataFrame) -> PredictionResult:
        """Generate predictions on test data.

        Parameters
        ----------
        test_data : pd.DataFrame
            Test data, same schema as train_data.

        Returns
        -------
        PredictionResult
            Predictions for SoH, SoC, RUL.
        """
        ...


# =============================================================================
# FEATURE COLUMNS (shared across tabular baselines)
# =============================================================================

# Columns used as input features for tabular models
FEATURE_COLS: list[str] = [
    "v_mean", "v_min", "v_max", "v_std", "v_slope",
    "i_mean", "i_std",
    "dq_dv_peak_v", "dq_dv_peak_mag",
    "v_ocv", "r0", "r1", "c1",
    "t_mean", "t_max", "t_std", "t_rise",
    "mean_temp_cycle", "c_rate",
    "rest_duration_hours", "soc_at_rest_onset", "ambient_temp_rest", "rest_flag",
]


def _prepare_features(df: pd.DataFrame) -> NDArray[np.float64]:
    """Extract feature matrix from DataFrame, handling NaN."""
    X = df[FEATURE_COLS].values.astype(np.float64)
    # Replace NaN with 0 (safe for tree models; linear models get imputed)
    X = np.nan_to_num(X, nan=0.0)
    return X


# =============================================================================
# 1. COULOMB COUNTING
# =============================================================================

class CoulombCountingModel(BaselineModel):
    """Open-loop Coulomb counting baseline.

    SoC_t = SoC_{t-1} + integral(I * dt) / Q_rated

    This is the simplest possible estimator. It drifts because there's no
    feedback correction (no observer). It only predicts SoC, not SoH or RUL.

    In our per-cycle features context, we approximate this by using the
    throughput_cycle_ah directly as a proxy for charge transferred.
    """

    name = "CoulombCounting"

    def __init__(self) -> None:
        self.rated_capacity: float = 2.0  # Ah

    def fit(self, train_data: pd.DataFrame, **hparams: Any) -> None:
        """No training needed -- Coulomb counting is model-free."""
        t0 = time.monotonic()
        # Nothing to fit -- store rated capacity from config
        self.rated_capacity = hparams.get("rated_capacity", 2.0)
        self._wall_clock = time.monotonic() - t0

    def predict(self, test_data: pd.DataFrame) -> PredictionResult:
        """Predict SoC via cumulative throughput integration.

        SoH: Coulomb counting cannot estimate SoH -- it only tracks SoC.
        RUL is not predicted (NaN).
        """
        n = len(test_data)

        # SoC: start at 1.0, subtract normalized throughput per cycle
        throughput = test_data["throughput_cycle_ah"].values
        soc = 1.0 - throughput / self.rated_capacity
        soc = np.clip(soc, 0.0, 1.0)

        # SoH: Coulomb counting has no capacity tracking mechanism.
        # Return NaN -- this baseline is purely a SoC estimator.
        soh = np.full(n, np.nan)

        # No RUL prediction
        rul = np.full(n, np.nan)

        return PredictionResult(
            soh=soh, soc=soc, rul=rul,
            wall_clock_seconds=self._wall_clock,
        )


# =============================================================================
# 2. EXTENDED KALMAN FILTER (1RC ECM)
# =============================================================================

class EKFModel(BaselineModel):
    """Extended Kalman Filter with 1RC equivalent circuit model.

    State: x = [SoC, V_p] (SoC and polarization voltage)
    Observation: y = V_terminal = OCV(SoC) - I*R0 - V_p

    The EKF uses the ECM parameters (R0, R1, C1) from data_loader.py
    as the system model. OCV is approximated as a linear function of SoC
    for simplicity (can be replaced with a lookup table).

    This is a classical model-based estimator, not a learning method.
    """

    name = "EKF_1RC"

    def __init__(self) -> None:
        self.r0: float = 0.1
        self.r1: float = 0.8
        self.c1: float = 1000.0
        self.rated_capacity: float = 2.0
        self.ocv_slope: float = 0.8  # dOCV/dSoC (linear approx)
        self.ocv_offset: float = 3.0  # OCV at SoC=0
        self._wall_clock: float = 0.0

    def _ocv(self, soc: float) -> float:
        """Linear OCV-SoC approximation: OCV = 3.0 + 0.8 * SoC."""
        return self.ocv_offset + self.ocv_slope * soc

    def fit(self, train_data: pd.DataFrame, **hparams: Any) -> None:
        """Calibrate ECM parameters from training data median values."""
        t0 = time.monotonic()

        # Use median ECM params from training data
        self.r0 = float(train_data["r0"].median())
        r1_vals = train_data["r1"].dropna()
        self.r1 = float(r1_vals.median()) if len(r1_vals) > 0 else 0.8
        c1_vals = train_data["c1"].dropna()
        self.c1 = float(c1_vals.median()) if len(c1_vals) > 0 else 1000.0
        self.rated_capacity = hparams.get("rated_capacity", 2.0)

        # Fit OCV curve from (SoC, V_ocv) pairs
        valid = train_data.dropna(subset=["v_ocv"])
        if len(valid) > 5:
            soc_vals = valid["soh"].values  # approximate SoC ~ SoH for this
            ocv_vals = valid["v_ocv"].values
            # Simple linear fit
            coeffs = np.polyfit(soc_vals, ocv_vals, 1)
            self.ocv_slope = float(coeffs[0])
            self.ocv_offset = float(coeffs[1])

        self._wall_clock = time.monotonic() - t0

    def predict(self, test_data: pd.DataFrame) -> PredictionResult:
        """Run EKF forward pass over test cycles."""
        n = len(test_data)

        # Initialize state
        soc = 1.0
        v_p = 0.0

        # Covariance matrices
        P = np.diag([0.01, 0.001])  # Initial state covariance
        Q = np.diag([1e-4, 1e-5])   # Process noise
        R_meas = np.array([[0.01]])  # Measurement noise

        soc_preds = np.zeros(n)
        soh_preds = np.zeros(n)

        for k in range(n):
            row = test_data.iloc[k]
            i_mean = float(row.get("i_mean", 0.0))
            v_meas = float(row.get("v_mean", 3.5))
            dt_cycle = float(row.get("throughput_cycle_ah", 0.0)) / max(abs(i_mean), 1e-6)

            tau = self.r1 * self.c1
            dt = max(dt_cycle, 1.0)  # seconds

            # State prediction
            soc_pred = soc - (abs(i_mean) * dt) / (self.rated_capacity * 3600)
            soc_pred = np.clip(soc_pred, 0.0, 1.0)
            alpha = np.exp(-dt / max(tau, 1.0))
            v_p_pred = alpha * v_p + self.r1 * (1 - alpha) * abs(i_mean)

            # Jacobian of state transition
            F = np.array([
                [1.0, 0.0],
                [0.0, alpha],
            ])

            # Predicted covariance
            P_pred = F @ P @ F.T + Q

            # Observation model: V = OCV(SoC) - I*R0 - V_p
            v_pred = self._ocv(soc_pred) - abs(i_mean) * self.r0 - v_p_pred

            # Jacobian of observation
            H = np.array([[self.ocv_slope, -1.0]])

            # Innovation
            y_innov = v_meas - v_pred

            # Kalman gain
            S = H @ P_pred @ H.T + R_meas
            K = P_pred @ H.T @ np.linalg.inv(S)

            # State update
            x_update = K @ np.array([[y_innov]])
            soc = float(soc_pred + x_update[0, 0])
            v_p = float(v_p_pred + x_update[1, 0])
            soc = np.clip(soc, 0.0, 1.0)

            # Covariance update
            P = (np.eye(2) - K @ H) @ P_pred

            soc_preds[k] = soc

            # SoH: Standard 1RC EKF is a SoC estimator, not a SoH estimator.
            # SoH estimation requires dual-EKF with capacity as a state variable,
            # which we haven't implemented here. Return NaN for honest comparison.
            soh_preds[k] = np.nan

        return PredictionResult(
            soh=soh_preds,
            soc=soc_preds,
            rul=np.full(n, np.nan),
            wall_clock_seconds=self._wall_clock,
        )


# =============================================================================
# 3. RANDOM FOREST
# =============================================================================

class RandomForestModel(BaselineModel):
    """Random Forest regressor on tabular cycle features.

    Separate RF models for SoH and RUL. It does not produce an intra-cycle
    SoC estimate, so its SoC prediction is explicitly NaN.
    """

    name = "RandomForest"

    def __init__(self) -> None:
        self._rf_soh = None
        self._rf_rul = None
        self._wall_clock: float = 0.0

    def fit(self, train_data: pd.DataFrame, **hparams: Any) -> None:
        """Train Random Forest models."""
        from sklearn.ensemble import RandomForestRegressor

        t0 = time.monotonic()

        X = _prepare_features(train_data)
        y_soh = train_data["soh"].values

        n_estimators = hparams.get("n_estimators", 100)
        max_depth = hparams.get("max_depth", 10)
        random_state = hparams.get("random_state", 42)

        self._rf_soh = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
        )
        self._rf_soh.fit(X, y_soh)

        # RUL: only fit on non-NaN labels
        rul_mask = ~np.isnan(train_data["rul"].values)
        if rul_mask.any():
            self._rf_rul = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=1,
            )
            self._rf_rul.fit(X[rul_mask], train_data["rul"].values[rul_mask])

        self._wall_clock = time.monotonic() - t0

    def predict(self, test_data: pd.DataFrame) -> PredictionResult:
        """Predict using trained RF models."""
        X = _prepare_features(test_data)

        soh = self._rf_soh.predict(X) if self._rf_soh else np.full(len(X), np.nan)
        soc = np.full(len(X), np.nan)  # RF is not an intra-cycle SoC estimator
        rul = self._rf_rul.predict(X) if self._rf_rul else np.full(len(X), np.nan)

        return PredictionResult(
            soh=soh, soc=soc, rul=rul,
            wall_clock_seconds=self._wall_clock,
        )


# =============================================================================
# 4. XGBOOST (with Optuna tuning)
# =============================================================================

class XGBoostModel(BaselineModel):
    """XGBoost regressor with optional Optuna hyperparameter tuning.

    If optuna_n_trials > 0 during fit(), runs a quick HPO search.
    """

    name = "XGBoost"

    def __init__(self) -> None:
        self._xgb_soh = None
        self._xgb_rul = None
        self._wall_clock: float = 0.0

    def fit(self, train_data: pd.DataFrame, **hparams: Any) -> None:
        """Train XGBoost models, optionally with Optuna tuning."""
        import xgboost as xgb

        t0 = time.monotonic()
        X = _prepare_features(train_data)
        y_soh = train_data["soh"].values

        n_trials = hparams.get("optuna_n_trials", 0)
        random_state = hparams.get("random_state", 42)

        if n_trials > 0:
            best_params = self._optuna_tune(X, y_soh, n_trials, random_state)
        else:
            best_params = {
                "n_estimators": hparams.get("n_estimators", 200),
                "max_depth": hparams.get("max_depth", 6),
                "learning_rate": hparams.get("learning_rate", 0.1),
                "subsample": hparams.get("subsample", 0.8),
                "colsample_bytree": hparams.get("colsample_bytree", 0.8),
            }

        self._xgb_soh = xgb.XGBRegressor(
            **best_params, random_state=random_state, verbosity=0,
        )
        self._xgb_soh.fit(X, y_soh)

        # RUL
        rul_mask = ~np.isnan(train_data["rul"].values)
        if rul_mask.any():
            self._xgb_rul = xgb.XGBRegressor(
                **best_params, random_state=random_state, verbosity=0,
            )
            self._xgb_rul.fit(X[rul_mask], train_data["rul"].values[rul_mask])

        self._wall_clock = time.monotonic() - t0

    def _optuna_tune(
        self,
        X: NDArray,
        y: NDArray,
        n_trials: int,
        random_state: int,
    ) -> dict[str, Any]:
        """Run Optuna HPO for XGBoost."""
        import optuna
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
            reg = xgb.XGBRegressor(**params, random_state=random_state, verbosity=0)
            scores = cross_val_score(reg, X, y, cv=3, scoring="neg_root_mean_squared_error")
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return study.best_params

    def predict(self, test_data: pd.DataFrame) -> PredictionResult:
        """Predict using trained XGBoost models."""
        X = _prepare_features(test_data)

        soh = self._xgb_soh.predict(X) if self._xgb_soh else np.full(len(X), np.nan)
        soc = np.full(len(X), np.nan)  # XGBoost is not an intra-cycle SoC estimator
        rul = self._xgb_rul.predict(X) if self._xgb_rul else np.full(len(X), np.nan)

        return PredictionResult(
            soh=soh, soc=soc, rul=rul,
            wall_clock_seconds=self._wall_clock,
        )


# =============================================================================
# 5. VANILLA LSTM
# =============================================================================

class VanillaLSTMModel(BaselineModel):
    """Bidirectional LSTM on multivariate time-series windows.

    Operates on sequences of cycle features. Uses the same tabular features
    as RF/XGBoost but treats them as a multivariate time series.
    """

    name = "VanillaLSTM"

    def __init__(self) -> None:
        self._model = None
        self._scaler = None
        self._wall_clock: float = 0.0
        self._device = "cpu"

    def fit(self, train_data: pd.DataFrame, **hparams: Any) -> None:
        """Train LSTM on windowed sequences."""
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        t0 = time.monotonic()

        random_state = hparams.get("random_state", 42)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        window_size = hparams.get("window_size", 10)
        hidden_dim = hparams.get("hidden_dim", 64)
        n_epochs = hparams.get("n_epochs", 50)
        lr = hparams.get("lr", 1e-3)
        batch_size = hparams.get("batch_size", 32)

        # Prepare features
        X_raw = _prepare_features(train_data)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_raw)

        y_soh = train_data["soh"].values
        y_soc = train_data["soc_eod"].values

        # Create windows
        X_windows, y_soh_w, y_soc_w = [], [], []
        for i in range(window_size, len(X_scaled)):
            X_windows.append(X_scaled[i - window_size:i])
            y_soh_w.append(y_soh[i])
            y_soc_w.append(y_soc[i])

        X_t = torch.tensor(np.array(X_windows), dtype=torch.float32)
        y_soh_t = torch.tensor(np.array(y_soh_w), dtype=torch.float32).unsqueeze(-1)
        y_soc_t = torch.tensor(np.array(y_soc_w), dtype=torch.float32).unsqueeze(-1)

        # Build model
        n_features = X_scaled.shape[1]
        self._model = _LSTMNet(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            output_dim=2,  # SoH + SoC
            bidirectional=True,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # Training loop
        self._model.train()
        dataset = torch.utils.data.TensorDataset(X_t, y_soh_t, y_soc_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
        )

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_X, batch_soh, batch_soc in loader:
                batch_X = batch_X.to(self._device)
                batch_y = torch.cat([batch_soh, batch_soc], dim=-1).to(self._device)

                optimizer.zero_grad()
                pred = self._model(batch_X)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        self._window_size = window_size
        self._wall_clock = time.monotonic() - t0

    def predict(self, test_data: pd.DataFrame) -> PredictionResult:
        """Predict using trained LSTM."""
        import torch

        n = len(test_data)
        X_raw = _prepare_features(test_data)

        if self._scaler is not None:
            X_scaled = self._scaler.transform(X_raw)
        else:
            X_scaled = X_raw

        soh_preds = np.full(n, np.nan)
        soc_preds = np.full(n, np.nan)

        if self._model is not None:
            self._model.eval()
            ws = self._window_size

            with torch.no_grad():
                for i in range(ws, n):
                    window = torch.tensor(
                        X_scaled[i - ws:i], dtype=torch.float32
                    ).unsqueeze(0).to(self._device)
                    pred = self._model(window).cpu().numpy()[0]
                    soh_preds[i] = pred[0]
                    soc_preds[i] = pred[1]

            # Fill early predictions with first valid
            for i in range(ws):
                soh_preds[i] = soh_preds[ws] if ws < n else np.nan
                soc_preds[i] = soc_preds[ws] if ws < n else np.nan

        return PredictionResult(
            soh=soh_preds,
            soc=soc_preds,
            rul=np.full(n, np.nan),
            wall_clock_seconds=self._wall_clock,
        )


class _LSTMNet(object):
    """Bidirectional LSTM network for time-series regression.

    Defined as a proper nn.Module but placed here to avoid circular imports.
    """

    def __new__(cls, input_dim: int, hidden_dim: int, output_dim: int,
                bidirectional: bool = True):
        import torch.nn as nn

        class LSTMNetModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=0.1,
                )
                mult = 2 if bidirectional else 1
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * mult, 32),
                    nn.GELU(),
                    nn.Linear(32, output_dim),
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last = lstm_out[:, -1, :]  # last time step
                return self.fc(last)

        return LSTMNetModule()


# =============================================================================
# 6. REST-TIME STATISTICAL DECOMPOSITION
# =============================================================================

class StatisticalDecompositionModel(BaselineModel):
    """Simplified Qin et al. RTPF: trend + regeneration decomposition.

    This is the primary rival baseline to SLAC. It uses statistical methods
    (not neural networks) to decompose SoH into:
    - A smooth monotonic trend (exponential or polynomial)
    - Regeneration bumps after rest events

    Approach:
    1. Identify rest-flagged cycles from rest_flag column
    2. Fit a smooth degradation trend on non-rest cycles
    3. Model regeneration amplitude as a function of rest duration
    4. Recombine: SoH_pred = trend(t) + regen(rest_features)

    This is the classical approach that SLAC aims to improve upon by making
    the decomposition differentiable and end-to-end trainable.
    """

    name = "StatDecomp"

    def __init__(self) -> None:
        self._trend_coeffs: NDArray | None = None
        self._regen_slope: float = 0.0
        self._regen_intercept: float = 0.0
        self._wall_clock: float = 0.0

    def fit(self, train_data: pd.DataFrame, **hparams: Any) -> None:
        """Fit degradation trend and regeneration model."""
        t0 = time.monotonic()

        cycle_idx = np.arange(len(train_data), dtype=np.float64)
        soh = train_data["soh"].values.astype(np.float64)
        rest_flag = train_data["rest_flag"].values.astype(int)

        # 1. Fit monotonic degradation trend on ALL cycles
        # Use polynomial (degree 2) for simplicity
        deg = hparams.get("trend_degree", 2)
        self._trend_coeffs = np.polyfit(cycle_idx, soh, deg)

        # 2. Model regeneration as residual on rest cycles
        trend_fitted = np.polyval(self._trend_coeffs, cycle_idx)
        residuals = soh - trend_fitted

        rest_mask = rest_flag == 1
        if rest_mask.any() and rest_mask.sum() >= 2:
            rest_durations = train_data.loc[rest_mask, "rest_duration_hours"].values
            rest_residuals = residuals[rest_mask]

            # Simple linear model: regen = slope * duration + intercept
            if len(rest_durations) >= 2:
                coeffs = np.polyfit(rest_durations, rest_residuals, 1)
                self._regen_slope = float(coeffs[0])
                self._regen_intercept = float(coeffs[1])
        else:
            self._regen_slope = 0.0
            self._regen_intercept = 0.0

        self._n_train = len(train_data)
        self._wall_clock = time.monotonic() - t0

    def predict(self, test_data: pd.DataFrame) -> PredictionResult:
        """Predict SoH using trend + regeneration decomposition."""
        n = len(test_data)
        # Continue cycle indexing from training
        cycle_idx = np.arange(self._n_train, self._n_train + n, dtype=np.float64)

        # Trend prediction
        soh_trend = np.polyval(self._trend_coeffs, cycle_idx)

        # Add regeneration on rest cycles
        rest_flag = test_data["rest_flag"].values.astype(int)
        rest_duration = test_data["rest_duration_hours"].values
        regen = np.where(
            rest_flag == 1,
            self._regen_slope * rest_duration + self._regen_intercept,
            0.0,
        )

        soh_pred = soh_trend + regen

        # SoC: not predicted by StatDecomp
        soc_pred = np.full(n, np.nan)

        # RUL: extrapolate trend to EOL threshold
        eol_soh = 0.70  # 30% fade from rated
        rul_pred = np.full(n, np.nan)
        if self._trend_coeffs is not None:
            # Find where trend crosses EOL
            trend_fn = np.poly1d(self._trend_coeffs)
            roots = (trend_fn - eol_soh).roots
            real_roots = roots[np.isreal(roots)].real
            future_roots = real_roots[real_roots > cycle_idx[0]]
            if len(future_roots) > 0:
                eol_cycle = float(np.min(future_roots))
                rul_pred = np.maximum(0, eol_cycle - cycle_idx)

        return PredictionResult(
            soh=soh_pred,
            soc=soc_pred,
            rul=rul_pred,
            wall_clock_seconds=self._wall_clock,
        )


# =============================================================================
# REGISTRY
# =============================================================================

BASELINE_REGISTRY: dict[str, type[BaselineModel]] = {
    "CoulombCounting": CoulombCountingModel,
    "EKF_1RC": EKFModel,
    "RandomForest": RandomForestModel,
    "XGBoost": XGBoostModel,
    "VanillaLSTM": VanillaLSTMModel,
    "StatDecomp": StatisticalDecompositionModel,
}
"""Registry of all baseline models. PI-MDCNet is in pi_mdcnet.py."""


def get_all_baselines() -> list[BaselineModel]:
    """Instantiate one of each baseline model."""
    return [cls() for cls in BASELINE_REGISTRY.values()]


# =============================================================================
# UNIT TESTS
# =============================================================================

def _run_baseline_tests() -> None:
    """Quick smoke tests for all baselines."""
    print("=" * 70)
    print("PI-MDCNet v4 -- Baseline Model Smoke Tests")
    print("=" * 70)

    # Create synthetic data matching data_loader output schema
    np.random.seed(42)
    n = 100
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
        "rest_duration_hours": np.concatenate([
            np.zeros(90), np.random.uniform(2, 20, 10)
        ]),
        "soc_at_rest_onset": np.random.uniform(0, 1, n),
        "ambient_temp_rest": np.random.uniform(22, 26, n),
        "rest_flag": np.concatenate([np.zeros(90), np.ones(10)]).astype(int),
        "soh": np.linspace(1.0, 0.7, n),
        "soc_eod": np.random.uniform(0.0, 0.1, n),
        "capacity_ah": np.linspace(2.0, 1.4, n),
        "rul": np.linspace(80, 0, n),
        "cycle_index": np.arange(n),
        "cell_id": ["B_TEST"] * n,
    })

    train_df = df.iloc[:80].copy()
    test_df = df.iloc[80:].copy()

    for name, cls in BASELINE_REGISTRY.items():
        print(f"\n-- {name} {'.' * (50 - len(name))}")
        try:
            model = cls()
            model.fit(train_df)
            result = model.predict(test_df)

            soh_valid = ~np.isnan(result.soh)
            soc_valid = ~np.isnan(result.soc)
            rul_valid = ~np.isnan(result.rul)

            print(f"  SoH: {soh_valid.sum()}/{len(result.soh)} valid, "
                  f"range=[{np.nanmin(result.soh):.4f}, {np.nanmax(result.soh):.4f}]")
            print(f"  SoC: {soc_valid.sum()}/{len(result.soc)} valid, "
                  f"range=[{np.nanmin(result.soc):.4f}, {np.nanmax(result.soc):.4f}]")
            print(f"  RUL: {rul_valid.sum()}/{len(result.rul)} valid"
                  + (f", range=[{np.nanmin(result.rul):.1f}, {np.nanmax(result.rul):.1f}]"
                     if rul_valid.any() else ""))
            print(f"  Wall-clock: {result.wall_clock_seconds:.3f}s")
            print(f"  [PASS]")
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Baseline smoke tests complete.")
    print("=" * 70)


if __name__ == "__main__":
    _run_baseline_tests()
