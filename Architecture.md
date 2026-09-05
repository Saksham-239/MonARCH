# System Architecture & Technical Specification

## 1. Overall System Architecture

The project consists of two tightly coupled environments:
1. **Local Benchmarking & Training Environment:** Rigorous, reproducible evaluation of physics-guided and baseline models against NASA PCoE battery data, producing an optimized, lightweight ONNX model graph.
2. **Serverless Cloud Fleet Analytics Pipeline (AWS Free-Tier):** An event-driven, zero-idle-cost architecture ingesting MQTT telemetry, orchestrating feature pipelines, running serverless inference, and serving a live monitoring dashboard.

### 1.1 Two-Timescale Battery State Estimation Architecture

The system explicitly decouples battery dynamics into two operational timescales:

#### A. Micro-Timescale: Intra-Cycle Continuous State of Charge (SoC)
* **Target:** Continuous intra-cycle state estimation $\text{SoC}(t) \in [0, 1]$.
* **Input Signals:** Fast intra-cycle time-series $[V(t), I(t), t]$.
* **Architecture:** Lightweight GRU sequence-to-sequence neural network (`IntraCycleSoCGRU`, 1 layer, hidden dim 32, sigmoid head).
* **Reference Truth:** Retrospective cycle-normalized SoC reference:
  $$\text{SoC}_{\text{ref}}(t) = 1.0 - \frac{q(t)}{Q_{\text{cycle}}} = 1.0 - \frac{\int_0^t |I(\tau)| d\tau}{\int_0^{T_{\text{end}}} |I(\tau)| d\tau}$$
  where the total cycle discharge capacity $Q_{\text{cycle}}$ is available retrospectively for evaluation/training, but is **strictly withheld from model inputs** during inference.

#### B. Macro-Timescale: Inter-Cycle Degradation & Regeneration (SLAC Core)
* **Target:** Cycle-by-cycle State of Health ($\text{SoH}_t$) and Remaining Useful Life ($\text{RUL}$).
* **Inputs:** Cycle electrical features $x_{\text{elec}}$ (13), thermal features $x_{\text{therm}}$ (4), stressor vector $s_t = [\text{mean\_temp\_cycle}, \text{c\_rate}]$ (2), rest features (3), rest flag, and normalized cycle index.
* **State Decomposition:**
  $$\text{SoH}_t = \text{SoH}_0 - D_t + R_t$$
* **Architecturally Monotonic Damage ($D_t$):**
  $$\Delta D_t = \lambda \cdot \text{Softplus}(f_\theta(s_t)) > 0, \quad D_t = D_{t-1} + \Delta D_t$$
  where $\lambda = 0.003 > 0$. Since $\text{Softplus}(u) = \ln(1 + e^u) > 0$ for all finite $u$, $\Delta D_t > 0$ strictly, guaranteeing $D_t > D_{t-1}$ by parametric construction rather than relying on a soft penalty loss.
* **Hard-Gated Reversible Relaxation ($R_t$):**
  $$R_t = \text{rest\_flag}_t \cdot R_{\text{head}}(\text{rest\_features}, h_t)$$
  guaranteeing $R_t \equiv 0.0$ on all active cycling cycles without rest.
* **Causal RUL Extrapolation:** Extrapolated directly from the monotonic latent damage trajectory $D_t \to D_{\text{fail}} = \text{SoH}_0 - 0.70$ without seeing future cycle data.

```
+--------------------------------------------------------------------------------------------------+
|                                    LOCAL BENCHMARKING ENGINE                                     |
|                                                                                                  |
|   NASA PCoE Datasets (B0005: 2.7V, B0006: 2.5V, B0007: 2.2V, B0018: 2.5V cutoff at 24°C)         |
|             |                                                                                    |
|             v                                                                                    |
|   [data_loader.py] ----> 1RC Thevenin Parameter Estimation + dQ/dV Spectrum + Rest Flags         |
|             |                                                                                    |
|             v                                                                                    |
|   [LOBO-CV Protocol] ---> Train on 3 cells, evaluate on 1 held-out cell                          |
|             |                                                                                    |
|             +---> Baselines (Coulomb Counting, EKF, RF, XGBoost, Vanilla LSTM)                   |
|             +---> PI-MDCNet v4 (SLAC Core: Parametric Monotonic D_t + Hard-Gated R_t)            |
|             +---> 4-Way Component Ablation Engine (Full, NoCrossAttn, NoHardGating, NoMono)       |
|             |                                                                                    |
|             v                                                                                    |
|   [export_onnx.py] ----> pi_mdcnet_edge.onnx (OpSet 14, ~318 KB, CPU-optimized)                 |
|   [embedded/esp32] ----> C99 Streaming Engine (ESP32-S3 / Xtensa LX7, 32-Byte State)             |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 | (Deploy model artifact to AWS Lambda)
                                                 v
+--------------------------------------------------------------------------------------------------+
|                               AWS SERVERLESS FLEET PIPELINE                                      |
|                                                                                                  |
|   [Multi-Cell Simulator] (5-10 virtual battery assets publishing V, I, T over MQTT)               |
|             |                                                                                    |
|             v                                                                                    |
|   AWS IoT Core (Topic: battery/+/telemetry)                                                      |
|             |                                                                                    |
|             +----(IoT SQL Rule Engine: SELECT * FROM 'battery/+/telemetry')                      |
|             |                                                                                    |
|             v                                                                                    |
|   AWS Lambda: Ingest Handler (Zero-buffer event ingestion)                                       |
|        |                                 |                                                       |
|        v (Hot Path)                      v (Cold Archive)                                        |
|   Amazon DynamoDB                 Amazon S3                                                      |
|   Table: `battery_telemetry`      Bucket: `battery-fleet-raw-archive`                            |
|        |                                                                                         |
|        v                                                                                         |
|   AWS Step Functions (Orchestrated Pipeline Execution)                                           |
|        |                                                                                         |
|        +---> [1] Feature Engineering Lambda: Compute rolling stats, dQ/dV, ECM priors            |
|        |         Writes to DynamoDB Table: `battery_features`                                    |
|        |                                                                                         |
|        +---> [2] Model Inference Lambda: Loads `pi_mdcnet_edge.onnx` via ONNX Runtime            |
|        |         Predicts: SoC (%), SoH (%), RUL (cycles), Anomaly Flag                          |
|        |         Writes to DynamoDB Table: `battery_predictions`                                 |
|        |                                                                                         |
|        +---> [3] Anomaly & Guardrail Check: If Temp > 45°C or dSoH > threshold                  |
|                  Triggers Amazon SNS Email Alert + IoT Device Shadow Throttle Command            |
|                                                                                                  |
|   Amazon API Gateway (REST API: GET /assets/{id}/status, GET /assets/{id}/history)               |
|             |                                                                                    |
|             v                                                                                    |
|   S3 Static Website / Amplify Hosting (Vanilla HTML5 + Modern CSS + Plotly.js Dashboard)        |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Technical Stack

| Domain | Technology / Library | Purpose / Rationale |
| :--- | :--- | :--- |
| **Language & Environment** | Python 3.12, `.venv` | Local runtime, isolated dependencies |
| **Deep Learning & Modeling**| PyTorch (`torch`) | Dynamic computational graphs, custom loss functions |
| **Classical ML & Baselines**| Scikit-learn, XGBoost | Rigorous baseline implementations |
| **Hyperparameter Tuning** | Optuna | Fair, budget-matched optimization across all models |
| **Scientific Computing** | NumPy, SciPy, Pandas | Signal processing, Savitzky-Golay filtering for $dQ/dV$ |
| **Inference Engine** | ONNX, ONNX Runtime (`onnxruntime`)| Ultra-fast CPU inference (< 20ms) inside AWS Lambda |
| **Data Visualization** | Matplotlib, Seaborn, Plotly.js | Publication-quality figures and interactive UI charts |
| **Cloud Ingestion** | AWS IoT Core (MQTT) | Low-overhead industrial IoT protocol |
| **Cloud Compute** | AWS Lambda (Python 3.12 runtime) | 100% serverless, zero idle billing |
| **Cloud Orchestration** | AWS Step Functions | Multi-stage pipeline coordination (< 4,000 free transitions) |
| **Cloud Storage** | Amazon DynamoDB, Amazon S3 | DynamoDB for low-latency state, S3 for cold archiving |
| **Cloud API & Serving** | Amazon API Gateway, S3 Hosting | Lightweight REST endpoints and static UI |
| **Alerting** | Amazon CloudWatch, Amazon SNS | Push notifications on thermal/degradation alarms |
| **Infrastructure as Code** | AWS CDK (TypeScript or Python) | Reproducible deployment and complete `cdk destroy` teardown |

---

## 3. Directory & File Structure

```
c:\Users\saksh\OneDrive\Documents\AWS ML\
├── PRD.md                         # Project Requirements Document
├── Architecture.md                # System Architecture & Technical Specification
├── Rules.md                       # AI Boundaries, Coding Standards, and Prohibitions
├── Phases.md                      # Milestone Breakdown & Implementation Phases
├── Design.md                      # UI/UX & Dashboard Visual Design System
├── requirements.txt               # Pinned Python dependencies
│
├── src/                           # Core Source Code
│   ├── __init__.py
│   ├── data_loader.py             # NASA PCoE parser, cycle feature extractor, rest detector
│   ├── ecm_thevenin.py            # 1-RC parameter estimation & Extended Kalman Filter (EKF)
│   ├── models/                    # Model Architecture Definitions
│   │   ├── __init__.py
│   │   ├── baselines.py           # Coulomb Counting, RF, XGBoost, Vanilla LSTM
│   │   ├── pi_mdcnet.py           # Split-Latent Aging Core (SLAC) Macro Architecture
│   │   └── intracycle_soc.py      # Auxiliary Intra-Cycle Continuous SoC GRU Module
├── embedded/                      # Embedded C Streaming Engine & ONNX Export
│   ├── export_onnx.py             # Single-cycle streaming ONNX export & sequential parity
│   └── esp32/                     # Pure C99 BMS state engine (ESP32-S3 / Xtensa LX7)
│       ├── CMakeLists.txt
│       └── main/
│           ├── pi_mdcnet_inference.h  # 32-byte persistent state definition & interface
│           ├── pi_mdcnet_inference.c  # Analytical streaming degradation/recovery closures
│           └── main.c                 # Standalone cycle-by-cycle simulation harness
│
├── experiments/                   # Benchmarking & Validation Scripts
│   ├── __init__.py
│   ├── train.py                   # LOBO cross-validation execution with unified training loop
│   ├── ablation.py                # 4-way component ablation experiment (Full, NoCrossAttn, NoHardGating, NoMono)
│   ├── rul_validation.py          # 4-way RUL comparative validation benchmark (Raw, StatDecomp, Neural, D_t Extrap)
│   ├── eval_rul_cell_breakdown.py # Per-cell RUL win/loss breakdown & degradation slope analysis
│   ├── eval_naive_soc.py          # Naive Coulomb counting zero-learning baseline
│   └── run_intracycle_soc.py      # LOBO evaluation of intra-cycle continuous SoC module
│
├── cloud/                         # Cloud Infrastructure & Lambda Functions
│   ├── lambdas/
│   │   ├── ingest_handler/        # Receives IoT MQTT events -> writes DynamoDB + S3
│   │   ├── feature_pipeline/      # Computes rolling window statistics & ECM features
│   │   ├── inference_engine/      # Runs ONNX runtime inference with packaged model
│   │   └── api_handler/           # Serves GET /assets/{id} for dashboard
│   ├── step_functions/
│   │   └── pipeline_state_machine.json # Step Functions definition
│   └── simulator/
│       └── fleet_simulator.py     # Multi-cell synthetic MQTT publisher
│
├── dashboard/                     # Static Web Frontend
│   ├── index.html                 # Single-page dashboard application
│   ├── styles.css                 # Industrial dark-mode CSS tokens
│   └── app.js                     # Plotly.js chart rendering and API Gateway polling
│
└── artifacts/                     # Generated Outputs, Metrics, and Models
    ├── comparison_table.csv       # LOBO-CV benchmark results (Mean ± Std)
    ├── ablation_table.csv         # Component ablation results (Full, NoCrossAttn, NoHardGating, NoMono)
    ├── rul_validation_results.csv # RUL benchmark breakdown across all folds & seeds
    ├── intracycle_soc_results.csv # Intra-cycle continuous SoC GRU benchmark results
    └── pi_mdcnet_edge.onnx        # Exported streaming ONNX model binary (~318 KB)
```

---

## 4. Hot Path vs. Cold Path Data Flow

1. **Hot Path (Operational Sub-second to Minute Scale):**
   - Telemetry ingested via IoT Core Rule $\rightarrow$ Ingest Lambda $\rightarrow$ PutItem into DynamoDB `battery_telemetry` table (TTL configured to 30 days to stay well within 25 GB free storage).
   - Step Functions invokes `feature_pipeline` and `inference_engine` every 5–15 minutes (or per cycle end).
   - Latest predictions written to DynamoDB `battery_predictions` table.
   - API Gateway reads directly from DynamoDB with single-digit millisecond latency to update the dashboard.

2. **Cold Path (Historical Archive & Retraining):**
   - Ingest Lambda batches raw JSON payloads into compressed S3 objects (`s3://battery-fleet-raw-archive/year=YYYY/month=MM/day=DD/asset_id/`).
   - S3 Lifecycle policy transitions data to Glacier or purges after demo runs.
   - Free SageMaker Studio Lab notebooks mount this S3 bucket for offline model retraining without provisioning cloud compute instances.
