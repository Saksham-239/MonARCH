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
2. **Financial Efficiency:** Full semester demonstration executed within AWS Free Tier and student credits ($0 net out-of-pocket).
3. **End-to-End Functionality:** Demonstrable closed loop: Simulated cell publishes MQTT telemetry $\rightarrow$ Cloud processes features and inference $\rightarrow$ Live dashboard updates with real-time health estimates.
