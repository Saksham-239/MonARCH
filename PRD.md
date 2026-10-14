# Project Requirements Document (PRD)

## Project Title
**Cloud-Orchestrated Multi-Domain Battery State Analytics for Intelligent Energy Assets**  
*(Free-Tier / Student Credit Edition)*

---

## 1. Executive Summary & Problem Statement
Lithium-ion battery packs in Electric Vehicles (EVs) and Grid Battery Energy Storage Systems (BESS) represent up to 40% of the total asset capital expenditure. Despite their criticality, commercial Battery Management Systems (BMS) operate with significant blind spots:
1. **Isolated Physical Domains:** Electrical ($V, I$), Thermal ($T$), and Aging/Usage dynamics are analyzed in silos, missing coupled failure modes like thermal-runaway precursors and temperature-accelerated lithium plating.
2. **Offline or Static Health Estimation:** Most commercial health indicators rely on periodic full-charge calibrations or coarse lookup tables, retiring packs based on calendar age rather than true electrochemical health (discarding packs with 20–30% useful life remaining).
3. **Prohibitive Cloud Ingestion Costs:** Enterprise cloud architectures often rely on always-on stream processors (e.g., Kinesis shards, managed real-time SageMaker endpoints) that incur continuous, steep hourly billing, making educational and small-fleet deployments economically unfeasible.

**This Project Solves This By:**
Building a fully serverless, zero-idle-cost cloud analytics pipeline on AWS Free Tier, driven by a rigorous, physics-guided multi-task machine learning model that co-estimates State of Charge (SoC), State of Health (SoH), and Remaining Useful Life (RUL) across coupled physical domains.

---

## 2. Target Users & Stakeholders
* **Battery Fleet Operators (EV / Microgrid):** Require real-time visibility into asset health, remaining cycle life, and automated safety alerts to schedule preventative maintenance before cell failure.
* **Energy Storage Engineers & Researchers:** Need high-fidelity data pipelines that accurately capture electrochemical relaxation and capacity fade without unphysical artifacts.
* **Academic Reviewers & Faculty:** Evaluate the project on:
  - Scientific rigor, fair baselines, and ablation studies (not unsubstantiated novelty claims).
  - Systems-level architectural judgment (cost-optimized serverless design).

---

## 3. Core Functional Requirements

### 3.1 Data Telemetry & Simulation
* **FR-1.1:** A multi-cell simulator script capable of publishing real-time telemetry (Voltage, Current, Cell Temperature, Cumulative Run-time) over MQTT to AWS IoT Core.
* **FR-1.2:** Simulation reflects real operational discharge-charge-rest profiles from standard NASA PCoE battery aging datasets with verified cell-specific discharge cutoff voltages: B0005 (2.7 V), B0006 (2.5 V), B0007 (2.2 V), and B0018 (2.5 V) at 24°C ambient temperature.

### 3.2 Physics-Informed Feature Engineering
* **FR-2.1:** Online extraction of Thevenin 1-RC equivalent circuit parameters ($V_{ocv}, R_0, R_1, C_1$).
* **FR-2.2:** Computation of Differential Capacity ($dQ/dV$) and Differential Voltage ($dV/dQ$) features during charging phases.
* **FR-2.3:** Tracking of preceding rest interval duration ($t_{\text{rest}}$) and impedance measurement flags to explicitly model electrochemical relaxation.

### 3.3 Two-Timescale Battery State Inference Architecture
* **FR-3.1:** Estimation of three key battery states across operational timescales:
  - **State of Charge (SoC(t)):** Real-time continuous state ($0 - 100\%$) estimated by an auxiliary intra-cycle micro-module (lightweight GRU) operating on continuous high-frequency $(V(t), I(t), t)$ telemetry. Evaluated against a retrospective cycle-normalized SoC reference derived from Coulomb integration using the measured discharge capacity of that cycle ($1 - q(t)/Q_{\text{cycle}}$), where $Q_{\text{cycle}}$ is strictly withheld from the model during inference.
  - **State of Health (SoH_t):** Macro-timescale per-cycle capacity fade decomposed via the Split-Latent Aging Core (SLAC) into irreversible monotonic damage ($D_t$) and gated reversible relaxation ($R_t$): $\text{SoH}_t = \text{SoH}_0 - D_t + R_t$.
  - **Remaining Useful Life (RUL):** Forecast of remaining operating cycles until End of Life ($70\%$ rated capacity = 1.40 Ah), evaluated via causal projection of the monotonic damage latent $D_t \to D_{\text{fail}}$ alongside neural and statistical extrapolation baselines. Cell B0007 is explicitly treated as right-censored (minimum capacity 1.4005 Ah > 1.40 Ah; never reaches EOL).
* **FR-3.2:** Model enforces architectural monotonicity on cumulative damage ($\Delta D_t = \lambda \cdot \text{Softplus}(f_\theta(s_t)) > 0$) while preserving valid rest-relaxation rebounds via hard gating ($R_t = \text{rest\_flag}_t \cdot R_{\text{head}}(\dots)$).

### 3.4 Serverless Cloud Orchestration
* **FR-4.1:** Direct event-driven ingestion from AWS IoT Core via SQL rules to AWS Lambda, storing hot-path records in Amazon DynamoDB and cold archives in Amazon S3.
* **FR-4.2:** AWS Step Functions state machine orchestrating periodic feature aggregation, batched model inference, and output persistence.
* **FR-4.3:** Automated safety alerts via Amazon SNS triggered when thermal gradients or degradation rates exceed safety thresholds.

### 3.5 Monitoring & Presentation
* **FR-5.1:** REST API via Amazon API Gateway providing telemetry, predictions, and asset status.
* **FR-5.2:** Responsive web dashboard (S3 static website / Amplify) displaying real-time gauges, time-series charts, and historical degradation curves.

---

## 4. Non-Functional & Operational Requirements

### 4.1 Cost Safety & Cloud Limits
* **NFR-1.1 (Zero Idle Cost):** The system must incur **$0.00/month** when idle. All services must be 100% serverless (Lambda, DynamoDB on-demand, S3, API Gateway).
* **NFR-1.2 (Strict Service Prohibitions):** Never deploy Kinesis Data Streams, AWS Glue DPUs, Amazon Timestream, or SageMaker Real-Time Endpoints.
* **NFR-1.3 (Budget Protection):** Mandatory AWS Budget alert at $1.00 and $5.00 thresholds.

### 4.2 Latency, Packaging & Embedded Memory Boundaries
* **NFR-2.1 (Cloud/Host Model):** The production model graph is compiled into an ONNX graph (`pi_mdcnet_edge.onnx`, 318,034 bytes / ~318 KB, opset 14, 67,013 float32 parameters), capable of running on CPU inside AWS Lambda with single-sample inference latency **< 20 ms**.
* **NFR-2.2 (Streaming BMS Persistent State):** Embedded C streaming state tracking operates with exactly **32 bytes** of persistent state (`pimdcnet_state_t`) and zero dynamic heap allocation, targeted for microcontrollers such as the ESP32-S3 (Xtensa LX7). (Note: 32 bytes refers strictly to persistent streaming state, not full on-device neural tensor inference).

### 4.3 Validation Rigor
* **NFR-3.1:** Local model benchmarking must use Leave-One-Battery-Out Cross-Validation (LOBO-CV) across all 4 NASA cells ($B0005, B0006, B0007, B0018$).
* **NFR-3.2:** All baselines (Coulomb Counting, EKF, Random Forest, XGBoost, Vanilla LSTM) must receive equal hyperparameter optimization budgets.
* **NFR-3.3:** Component ablations must isolate the specific contributions of architectural monotonicity, hard gating, and cross-attention.

---

## 5. Success Metrics
1. **Academic Defensibility:** All claims backed by LOBO-CV benchmark tables (mean $\pm$ std) and complete component ablations. With $N=4$ cells ($N=3$ valid EOL cells for RUL), formal statistical significance ($p < 0.05$) cannot be established (minimum two-sided Wilcoxon $p = 0.125$ for $N=4$ and $p = 0.250$ for $N=3$); all inferential statistics are reported as exploratory/descriptive only.
2. **Financial Efficiency:** Full semesteYou are acting as a Principal ML Systems Engineer and Lead Peer Reviewer for IEEE Transactions on Industrial Informatics / Applied Energy. Your objective is to audit, refactor, and upgrade this repository (`MonARCH` / `PI-MDCNet v4`) into a rock-solid, zero-leakage, publication-grade codebase ready for large-scale AWS benchmarking and paper submission.

Execute the following 6 phases across the exact files in this repository. Do not use mock results, hardcoded metrics, or placeholder C stubs.

### Phase 1: Centralize Experimental Protocol & Eliminate Leakage (`experiments/protocol.py` & `src/data_loader.py`)
1. **`experiments/protocol.py`**: Currently this file is nearly empty. Build a strict, centralized `ExperimentProtocol` dataclass and configuration registry inside `experiments/protocol.py` that defines:
   - Deterministic seed list: `SEEDS = [42, 100, 2024, 2025, 2026]`
   - End-of-Life (EOL) thresholds (e.g., 70% / 80% nominal capacity)
   - Window sizes, prediction horizons, and Leave-One-Cell-Out (LOCO) cross-validation folds
   - Standard evaluation metrics (`RMSE`, `MAE`, `MAPE`, `R2`, `RUL_AE`, `Alpha_Lambda_Accuracy`)
   Import and enforce `experiments/protocol.py` across `experiments/train.py`, `experiments/evaluate.py`, `experiments/ablation.py`, `experiments/rul_validation.py`, and `experiments/eval_rul_cell_breakdown.py`.
2. **Causal SLAC Decomposition (`src/data_loader.py`)**: Audit the SLAC decomposition in `src/data_loader.py`. Ensure that decomposing capacity into irreversible trend + local regeneration components at cycle `t` NEVER accesses capacity measurements from `t+1 ... N` (no two-sided savgol/wavelet/smoothing filters across the full trajectory prior to splitting). Implement a strictly causal expanding/rolling SLAC operator and add a unit test in `tests/test_data_loader.py` proving `SLAC(capacity[:t])[-1] == SLAC(capacity[:t+k])[t-1]` for all `k > 0`.

### Phase 2: Multi-Dataset Expansion (`scripts/download_data.py` & `src/data_loader.py`)
Expand beyond NASA `B0005.mat` so the paper demonstrates cross-cell and cross-chemistry generalization:
1. **`scripts/download_data.py`**: Upgrade to automatically download, checksum-verify, and cache:
   - **NASA PCoE**: Full 24°C quad (`B0005`, `B0006`, `B0007`, `B0018`).
   - **CALCE (University of Maryland)**: `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38` (prismatic LCO cells, 800+ cycles with strong rest-period capacity regeneration).
   - **Oxford Battery Degradation Dataset** (Kokam pouch cells `Cell1`–`Cell8` under dynamic urban Artemis discharge profiles) for dynamic intracycle SOC + SOH validation.
2. **`src/data_loader.py`**: Refactor into modular dataset adapters (`NASAAdapter`, `CALCEAdapter`, `OxfordAdapter`) outputting a unified tensor schema `(cycle_features, slac_trend, slac_regen, soh_target, rul_target, intracycle_voltage_current_temp)`.

### Phase 3: Physics-Informed Loss & Coupled Intracycle SOC (`src/models/pi_mdcnet.py` & `src/models/intracycle_soc.py`)
1. **`src/models/pi_mdcnet.py`**: Audit the Physics-Informed loss terms. Ensure the monotonic degradation constraint (`d(SOH)/dt <= 0`) is applied **exclusively to the SLAC irreversible trend head**, while the regeneration head is governed by rest-time/relaxation physics so the physics loss does not suppress legitimate capacity regeneration spikes.
2. **`src/models/intracycle_soc.py`, `experiments/run_intracycle_soc.py`, & `experiments/eval_naive_soc.py`**: Couple intracycle SOC estimation directly to `PI-MDCNet`'s predicted SOH (`Q_aged = SOH_pred * Q_nominal`). Show quantitatively in `artifacts/intracycle_soc_results.csv` how naive Coulomb counting / OCV lookup (`eval_naive_soc.py` assuming static `Q_nominal`) drifts severely as the battery ages (95%, 85%, 75% SOH), whereas SOH-conditioned `intracycle_soc.py` maintains <1.5% SOC RMSE across aged cycles on both NASA and Oxford dynamic profiles.

### Phase 4: Fix Duplicate CSV Artifacts & Multi-Seed Statistical Rigor (`experiments/evaluate.py` & `experiments/ablation.py`)
Currently `artifacts/ablation_results.csv` is byte-identical to `artifacts/ablation_table.csv`, and `artifacts/benchmark_results.csv` is byte-identical to `artifacts/comparison_table.csv`.
1. Refactor `experiments/evaluate.py`, `experiments/ablation.py`, `experiments/rul_validation.py`, and `experiments/eval_rul_cell_breakdown.py` to execute across all 5 seeds in `experiments/protocol.py` and across all cells in Leave-One-Cell-Out validation.
2. Save granular per-seed, per-cell rows to `artifacts/*_results.csv`.
3. Aggregate `Mean ± Std` across seeds, compute paired Wilcoxon signed-rank / Student's t-test p-values against `src/models/baselines.py` (MLP, LSTM/BiLSTM, TCN, Transformer, Vanilla PINN), and output formatted summary tables to `artifacts/comparison_table.csv`, `artifacts/ablation_table.csv`, and `artifacts/paper_tables.tex`.

### Phase 5: Real ESP32 C Inference & Quantization (`embedded/export_onnx.py` & `embedded/esp32/main/`)
Make the embedded BMS contribution genuinely publication-grade:
1. **`embedded/export_onnx.py`**: Export both FP32 (`pi_mdcnet_edge.onnx`) and INT8 quantized (`pi_mdcnet_edge_int8.onnx`) models, AND auto-generate a C header file `embedded/esp32/main/pi_mdcnet_weights.h` containing the exported quantized/FP32 weight tensors and dilation config.
2. **`embedded/esp32/main/pi_mdcnet_inference.h` & `pi_mdcnet_inference.c`**: Implement the complete, dependency-free C99 forward pass of `PI-MDCNet v4` (causal dilated 1D convolutions, activations, SLAC trend/regeneration fusion head, and static arena buffer allocation with zero `malloc`).
3. **`embedded/esp32/main/main.c`**: Implement a deterministic cycle-count (`esp_cpu_get_cycle_count()`) latency & SRAM profiler that runs a test vector and prints exact MACs, Flash size (KB), static SRAM footprint (Bytes), and latency (us).
4. Add a Python C-binding / subprocess parity test in `tests/test_pi_mdcnet.py` verifying that `pi_mdcnet_inference.c` matches PyTorch/`onnxruntime` outputs within `< 1e-4` (FP32) / `< 1e-2` (INT8).

### Phase 6: Repo Hygiene, Comprehensive Pytest Suite & Publication Figures
1. Delete `artifacts_archive/` and move internal agent tracking files (`Memory.md`, `Phases.md`, `Rules.md`, `PRD.md`, `artifacts/forensic_verification_report.md`) to `.internal_docs/` (added to `.gitignore`). Update `.gitignore` to exclude `data/raw/*.mat`.
2. Expand `tests/test_data_loader.py`, `tests/test_evaluate.py`, `tests/test_baselines.py`, and `tests/test_pi_mdcnet.py` to achieve >90% coverage of core invariants (zero causal leakage, disjoint cell splits, gradient flow, ONNX/C numerical parity).
3. Update figure generation to save high-resolution vector `.pdf` and 300-DPI `.png` figures with `±1 std` shaded bands across seeds.You are acting as a Principal ML Systems Engineer and Lead Peer Reviewer for IEEE Transactions on Industrial Informatics / Applied Energy. Your objective is to audit, refactor, and upgrade this repository (`MonARCH` / `PI-MDCNet v4`) into a rock-solid, zero-leakage, publication-grade codebase ready for large-scale AWS benchmarking and paper submission.

Execute the following 6 phases across the exact files in this repository. Do not use mock results, hardcoded metrics, or placeholder C stubs.

### Phase 1: Centralize Experimental Protocol & Eliminate Leakage (`experiments/protocol.py` & `src/data_loader.py`)
1. **`experiments/protocol.py`**: Currently this file is nearly empty. Build a strict, centralized `ExperimentProtocol` dataclass and configuration registry inside `experiments/protocol.py` that defines:
   - Deterministic seed list: `SEEDS = [42, 100, 2024, 2025, 2026]`
   - End-of-Life (EOL) thresholds (e.g., 70% / 80% nominal capacity)
   - Window sizes, prediction horizons, and Leave-One-Cell-Out (LOCO) cross-validation folds
   - Standard evaluation metrics (`RMSE`, `MAE`, `MAPE`, `R2`, `RUL_AE`, `Alpha_Lambda_Accuracy`)
   Import and enforce `experiments/protocol.py` across `experiments/train.py`, `experiments/evaluate.py`, `experiments/ablation.py`, `experiments/rul_validation.py`, and `experiments/eval_rul_cell_breakdown.py`.
2. **Causal SLAC Decomposition (`src/data_loader.py`)**: Audit the SLAC decomposition in `src/data_loader.py`. Ensure that decomposing capacity into irreversible trend + local regeneration components at cycle `t` NEVER accesses capacity measurements from `t+1 ... N` (no two-sided savgol/wavelet/smoothing filters across the full trajectory prior to splitting). Implement a strictly causal expanding/rolling SLAC operator and add a unit test in `tests/test_data_loader.py` proving `SLAC(capacity[:t])[-1] == SLAC(capacity[:t+k])[t-1]` for all `k > 0`.

### Phase 2: Multi-Dataset Expansion (`scripts/download_data.py` & `src/data_loader.py`)
Expand beyond NASA `B0005.mat` so the paper demonstrates cross-cell and cross-chemistry generalization:
1. **`scripts/download_data.py`**: Upgrade to automatically download, checksum-verify, and cache:
   - **NASA PCoE**: Full 24°C quad (`B0005`, `B0006`, `B0007`, `B0018`).
   - **CALCE (University of Maryland)**: `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38` (prismatic LCO cells, 800+ cycles with strong rest-period capacity regeneration).
   - **Oxford Battery Degradation Dataset** (Kokam pouch cells `Cell1`–`Cell8` under dynamic urban Artemis discharge profiles) for dynamic intracycle SOC + SOH validation.
2. **`src/data_loader.py`**: Refactor into modular dataset adapters (`NASAAdapter`, `CALCEAdapter`, `OxfordAdapter`) outputting a unified tensor schema `(cycle_features, slac_trend, slac_regen, soh_target, rul_target, intracycle_voltage_current_temp)`.

### Phase 3: Physics-Informed Loss & Coupled Intracycle SOC (`src/models/pi_mdcnet.py` & `src/models/intracycle_soc.py`)
1. **`src/models/pi_mdcnet.py`**: Audit the Physics-Informed loss terms. Ensure the monotonic degradation constraint (`d(SOH)/dt <= 0`) is applied **exclusively to the SLAC irreversible trend head**, while the regeneration head is governed by rest-time/relaxation physics so the physics loss does not suppress legitimate capacity regeneration spikes.
2. **`src/models/intracycle_soc.py`, `experiments/run_intracycle_soc.py`, & `experiments/eval_naive_soc.py`**: Couple intracycle SOC estimation directly to `PI-MDCNet`'s predicted SOH (`Q_aged = SOH_pred * Q_nominal`). Show quantitatively in `artifacts/intracycle_soc_results.csv` how naive Coulomb counting / OCV lookup (`eval_naive_soc.py` assuming static `Q_nominal`) drifts severely as the battery ages (95%, 85%, 75% SOH), whereas SOH-conditioned `intracycle_soc.py` maintains <1.5% SOC RMSE across aged cycles on both NASA and Oxford dynamic profiles.

### Phase 4: Fix Duplicate CSV Artifacts & Multi-Seed Statistical Rigor (`experiments/evaluate.py` & `experiments/ablation.py`)
Currently `artifacts/ablation_results.csv` is byte-identical to `artifacts/ablation_table.csv`, and `artifacts/benchmark_results.csv` is byte-identical to `artifacts/comparison_table.csv`.
1. Refactor `experiments/evaluate.py`, `experiments/ablation.py`, `experiments/rul_validation.py`, and `experiments/eval_rul_cell_breakdown.py` to execute across all 5 seeds in `experiments/protocol.py` and across all cells in Leave-One-Cell-Out validation.
2. Save granular per-seed, per-cell rows to `artifacts/*_results.csv`.
3. Aggregate `Mean ± Std` across seeds, compute paired Wilcoxon signed-rank / Student's t-test p-values against `src/models/baselines.py` (MLP, LSTM/BiLSTM, TCN, Transformer, Vanilla PINN), and output formatted summary tables to `artifacts/comparison_table.csv`, `artifacts/ablation_table.csv`, and `artifacts/paper_tables.tex`.

### Phase 5: Real ESP32 C Inference & Quantization (`embedded/export_onnx.py` & `embedded/esp32/main/`)
Make the embedded BMS contribution genuinely publication-grade:
1. **`embedded/export_onnx.py`**: Export both FP32 (`pi_mdcnet_edge.onnx`) and INT8 quantized (`pi_mdcnet_edge_int8.onnx`) models, AND auto-generate a C header file `embedded/esp32/main/pi_mdcnet_weights.h` containing the exported quantized/FP32 weight tensors and dilation config.
2. **`embedded/esp32/main/pi_mdcnet_inference.h` & `pi_mdcnet_inference.c`**: Implement the complete, dependency-free C99 forward pass of `PI-MDCNet v4` (causal dilated 1D convolutions, activations, SLAC trend/regeneration fusion head, and static arena buffer allocation with zero `malloc`).
3. **`embedded/esp32/main/main.c`**: Implement a deterministic cycle-count (`esp_cpu_get_cycle_count()`) latency & SRAM profiler that runs a test vector and prints exact MACs, Flash size (KB), static SRAM footprint (Bytes), and latency (us).
4. Add a Python C-binding / subprocess parity test in `tests/test_pi_mdcnet.py` verifying that `pi_mdcnet_inference.c` matches PyTorch/`onnxruntime` outputs within `< 1e-4` (FP32) / `< 1e-2` (INT8).

### Phase 6: Repo Hygiene, Comprehensive Pytest Suite & Publication Figures
1. Delete `artifacts_archive/` and move internal agent tracking files (`Memory.md`, `Phases.md`, `Rules.md`, `PRD.md`, `artifacts/forensic_verification_report.md`) to `.internal_docs/` (added to `.gitignore`). Update `.gitignore` to exclude `data/raw/*.mat`.
2. Expand `tests/test_data_loader.py`, `tests/test_evaluate.py`, `tests/test_baselines.py`, and `tests/test_pi_mdcnet.py` to achieve >90% coverage of core invariants (zero causal leakage, disjoint cell splits, gradient flow, ONNX/C numerical parity).
3. Update figure generation to save high-resolution vector `.pdf` and 300-DPI `.png` figures with `±1 std` shaded bands across seeds.r demonstration executed within AWS Free Tier and student credits ($0 net out-of-pocket).
3. **End-to-End Functionality:** Demonstrable closed loop: Simulated cell publishes MQTT telemetry $\rightarrow$ Cloud processes features and inference $\rightarrow$ Live dashboard updates with real-time health estimates.
