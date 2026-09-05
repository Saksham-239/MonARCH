# Project Implementation Phases & Milestones (Phases.md)

## Phase Overview
To ensure methodical development, validation, and cost safety, the project is structured into **10 sequential phases**. Each phase contains clear deliverables and automated verification criteria.

```
[Phase 0: Environment Setup] ──> [Phase 1: Data Ingestion & Features] ──> [Phase 2: Baseline Models]
                                                                                     │
[Phase 5: Metrics & Ablations] <── [Phase 4: LOBO Benchmark] <── [Phase 3: Hybrid PI-MDCNet]
        │
        └──> [Phase 6: ONNX Compilation] ──> [Phase 7: Cloud Pipeline] ──> [Phase 8: Web Dashboard]
                                                                                     │
                                                                       [Phase 9: Final Review & Docs]
```

---

## Phase 0: Local Environment & Dependency Isolation
* **Objective:** Establish an isolated, reproducible local Python 3.12 environment with locked dependencies.
* **Key Tasks:**
  1. Initialize local virtual environment (`.venv`).
  2. Install core packages (`torch`, `xgboost`, `scikit-learn`, `optuna`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `onnx`, `onnxruntime`).
  3. Freeze environment to `requirements.txt`.
* **Verification:** Run a sanity script confirming PyTorch, XGBoost, and ONNX Runtime versions.

---

## Phase 1: NASA PCoE Ingestion & Multi-Domain Feature Pipeline
* **Objective:** Build a robust data ingestion module that loads NASA PCoE battery aging cycles ($B0005, B0006, B0007, B0018$) and extracts electrochemical features.
* **Key Tasks:**
  1. Parse discharge, charge, and impedance cycle data.
  2. Implement online 1-RC Thevenin parameter extraction ($V_{ocv}, R_0, R_1, C_1$).
  3. Compute Differential Capacity ($dQ/dV$) curves via Savitzky-Golay filtering.
  4. Extract preceding rest times ($t_{\text{rest}}$) and mark electrochemical relaxation cycles.
  5. Structure the 4 Leave-One-Battery-Out (LOBO-CV) cross-validation folds.
* **Verification:** Execute `python -m src.data_loader` to verify clean cycle arrays, correct rest duration flags, and zero data leakage across LOBO folds.

---

## Phase 2: Fairly-Tuned Baseline Implementations
* **Objective:** Implement and optimize the standard industry baselines across physics, classical ML, and deep learning paradigms.
* **Key Tasks:**
  1. **Coulomb Counting:** Open-loop current integration with sensor bias modeling.
  2. **Extended Kalman Filter (EKF):** Closed-loop 1-RC state estimator tracking $SoC$ and polarization voltage $V_p$.
  3. **Random Forest Regressor:** Multi-output regression on tabular cycle summary features.
  4. **XGBoost Regressor:** Gradient-boosted decision trees with Optuna hyperparameter tuning.
  5. **Vanilla LSTM:** Bidirectional recurrent network on raw multi-variate time-series windows.
* **Verification:** Train each baseline on Fold 1 and confirm convergence and non-trivial prediction errors.

---

## Phase 3: Split-Latent Aging Core (SLAC) & Multi-Timescale Inference
* **Objective:** Implement the dual-timescale architecture with parametric monotonic damage increments and hard-gated relaxation.
* **Key Tasks:**
  1. Construct the macro-timescale SLAC core:
     $$\text{SoH}_t = \text{SoH}_0 - D_t + R_t$$
     $$\Delta D_t = \lambda \cdot \text{Softplus}(f_\theta(s_t)) > 0, \quad D_t = D_{t-1} + \Delta D_t$$
     $$R_t = \text{rest\_flag}_t \cdot R_{\text{head}}(\text{rest\_features}, h_t)$$
  2. Implement cross-attention fusion between electrical ($x_{\text{elec}}$) and thermal ($x_{\text{therm}}$) embeddings.
  3. Implement the auxiliary intra-cycle micro-module (continuous SoC GRU) operating on $[V(t), I(t), t]$.
  4. Code combined loss with masked right-censored RUL and L1 relaxation sparsity.
* **Verification:** Run `pytest tests/test_pi_mdcnet.py` to confirm gradient flow, exact decomposition, strict monotonicity of $D_t$, and hard gating of $R_t$.

---

## Phase 4: Leave-One-Battery-Out (LOBO) Benchmark Execution
* **Objective:** Execute the multi-seed, 4-fold LOBO cross-validation protocol comparing all models.
* **Key Tasks:**
  1. Run 4 LOBO-CV folds across all models (Coulomb Counting, EKF, RF, XGBoost, Vanilla LSTM, PI-MDCNet).
  2. Repeat across benchmark seeds to compute mean and standard deviation.
  3. Record per-fold metrics: SoH RMSE/MAE, SoC RMSE/MAE, RUL RMSE/MAE (with B0007 right-censored), and wall-clock training time.
* **Verification:** Generate `artifacts/comparison_table.csv` and verify reproducible cross-validation results.

---

## Phase 5: Component Ablation Study & Causal RUL Validation
* **Objective:** Rigorously isolate which components drive model performance and evaluate RUL extrapolation.
* **Key Tasks:**
  1. Evaluate 4 model variants across 4 LOBO folds and 3 seeds:
     - `Full_SLAC`
     - `Ablation_NoCrossAttn` (modality concatenation instead of cross-attention)
     - `Ablation_NoHardGating` (unconstrained relaxation prediction)
     - `Ablation_NoMonotonicity` (unconstrained cumulative damage increment)
  2. Execute comparative RUL benchmark (`Raw_SoH_Extrap`, `Stat_Decomp`, `Neural_RULHead`, `Latent_Dt_Extrap`).
  3. Analyze degradation acceleration ratios (B0005 knee at 1.67× vs B0006 at 0.42× vs B0018 at 0.68×).
* **Verification:** Generate `artifacts/ablation_table.csv` and `artifacts/rul_validation_results.csv`.

---

## Phase 6: ONNX Compilation & Embedded BMS State Engine
* **Objective:** Compile the streaming model to ONNX and demonstrate the embedded BMS streaming state engine.
* **Key Tasks:**
  1. Export streaming single-cycle model graph to `artifacts/pi_mdcnet_edge.onnx` (~318 KB).
  2. Validate numerical parity ($|\text{PyTorch} - \text{ONNX}| < 10^{-5}$) on both single steps and 50-step sequential state transitions.
  3. Implement and compile pure C99 streaming BMS state engine (`embedded/esp32/`) with verified 32-byte persistent state (`pimdcnet_state_t`) and zero heap allocation on ESP32-S3 (Xtensa LX7).
* **Verification:** Run `python -m embedded.export_onnx` and `.\embedded\esp32_demo.exe`.

---

## Phase 7: AWS Serverless Cloud Pipeline (Free-Tier Stack)
* **Objective:** Deploy the zero-idle-cost cloud ingestion and orchestration pipeline using AWS CDK / CloudFormation.
* **Key Tasks:**
  1. Configure AWS IoT Core with topic `battery/{asset_id}/telemetry` and direct SQL rule.
  2. Provision DynamoDB tables (`battery_telemetry`, `battery_features`, `battery_predictions`) in On-Demand mode.
  3. Package and deploy AWS Lambda functions (Ingest Handler, Feature Pipeline, ONNX Inference Engine, API Handler).
  4. Deploy AWS Step Functions state machine to coordinate pipeline execution.
  5. Configure CloudWatch alarms and Amazon SNS email notifications.
* **Verification:** Trigger a synthetic payload from IoT Core console and verify end-to-end data population in DynamoDB.

---

## Phase 8: Multi-Asset Simulator & Live Web Dashboard
* **Objective:** Deploy the multi-cell MQTT fleet simulator and interactive browser dashboard.
* **Key Tasks:**
  1. Implement `cloud/simulator/fleet_simulator.py` simulating 5 virtual battery assets (EV, BESS, Consumer).
  2. Build single-page static dashboard in `dashboard/` with Plotly.js charts and live polling of API Gateway.
  3. Host on Amazon S3 Static Website hosting or AWS Amplify free tier.
* **Verification:** Run simulator for 10 minutes and verify live real-time chart updates on the dashboard.

---

## Phase 9: Teardown Testing, Final Report & Presentation Alignment
* **Objective:** Validate cost safety and prepare academic defense materials.
* **Key Tasks:**
  1. Execute `cdk destroy` to verify that 100% of cloud resources can be torn down cleanly between sessions.
  2. Verify that AWS Billing shows **$0.00** charges.
  3. Update project presentation slides and technical documentation to reflect empirical findings, the ablation study, and the defensible systems framing.
* **Verification:** Clean terminal run, verified zero billing, and completed slide deck.
