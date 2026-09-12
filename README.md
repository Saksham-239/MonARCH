# MonARCH — Physics-Guided Battery State & RUL Prediction

**MonARCH** (*Monotonic Aging and Recovery-Constrained Neural State-Space Network*) is a physics-guided deep-learning framework for estimating **State of Charge (SoC), State of Health (SoH), and Remaining Useful Life (RUL)** of lithium-ion batteries.

The core idea is simple: instead of asking a neural network to learn physically implausible battery behaviour and then punishing it with loss terms, important constraints are built directly into the architecture.

> **MonARCH = learned degradation + physically constrained recovery + causal RUL extrapolation + edge deployment.**

## Why This Project?

Battery health is not a single-timescale problem. Fast electrical behaviour within a cycle and slow degradation across hundreds of cycles contain different information.

MonARCH therefore separates them:

```text
                 Lithium-ion battery telemetry
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Intra-cycle                  Inter-cycle
          dynamics                   degradation
             │                           │
             ▼                           ▼
       SoC estimation              SLAC core
                                         │
                           ┌─────────────┴─────────────┐
                           ▼                           ▼
                    Irreversible D(t)          Recovery R(t)
                    monotonic by design         hard-gated by rest
                           │                           │
                           └─────────────┬─────────────┘
                                         ▼
                              SoH = SoH₀ − D + R
                                         │
                                         ▼
                              Causal RUL extrapolation
```

## Core Contributions

### 1. Architecturally monotonic degradation

The cumulative damage state is updated as:

```text
ΔD_t = λ * Softplus(f_θ(s_t))
D_t  = D_(t-1) + ΔD_t
```

Because the Softplus increment is strictly non-negative (`ΔD_t >= 0`) by construction, the cumulative damage trajectory cannot spontaneously decrease, ensuring physical degradation monotonicity across battery life.

### 2. Hard-gated recovery

Battery relaxation after qualifying rest periods can produce apparent capacity recovery. MonARCH models this separately using a physical gating mechanism:

```text
R_t = rest_flag_t * R_head(rest_features, h_t)
```

If there was no qualifying rest event (`rest_flag_t == 0`), the recovery branch is structurally forced to zero, preventing the network from hallucinating unphysical capacity jumps during active cycling.

### 3. Two-timescale estimation

- **Micro-timescale:** a lightweight GRU estimates continuous intra-cycle SoC from voltage/current/time sequences.
- **Macro-timescale:** the Split-Latent Aging Core (SLAC) tracks degradation and recovery across cycles.

### 4. Causal RUL prediction

RUL is extrapolated from the learned monotonic damage trajectory toward an explicit end-of-life threshold rather than using future battery observations as hidden information.

### 5. Edge-oriented implementation

The repository includes an exported ONNX model and a compact C99 streaming implementation targeting the **ESP32-S3 / Xtensa LX7**. The embedded state is designed around a **32-byte persistent state footprint**.

## Evaluation Strategy

The project uses **Leave-One-Battery-Out (LOBO) cross-validation** on NASA PCoE battery-aging data: train on three cells and evaluate on the held-out cell.

It also includes comparisons against classical and neural baselines:

- Coulomb counting
- Extended Kalman Filter (EKF)
- Random Forest
- XGBoost
- Vanilla LSTM
- MonARCH / SLAC

Architectural ablations test the contribution of:

- Cross-attention
- Hard recovery gating
- Monotonic damage construction

The repository also contains RUL validation, intra-cycle SoC evaluation, benchmark tables, and a forensic verification report.

## System Architecture

```text
NASA PCoE battery data
        │
        ▼
Feature extraction
├── electrical statistics
├── thermal statistics
├── dQ/dV features
├── 1-RC Thevenin parameters
└── rest-event features
        │
        ▼
┌──────────────────────────────────────────┐
│              MonARCH / SLAC              │
│                                          │
│ Electrical Encoder ─┐                    │
│                     ├─► Cross Attention  │
│ Thermal Encoder ────┘         │          │
│                               ├─► SoH    │
│ Stressor ─► Monotonic D(t) ───┤          │
│ Rest ─────► Gated R(t) ───────┤          │
│                               ├─► RUL    │
└───────────────────────────────┴──────────┘
        │
        ├──► Benchmarking & ablation
        ├──► ONNX export
        └──► ESP32-S3 C99 engine
```

## Repository Structure

```text
.
├── Architecture.md             # Detailed architecture & system specification
├── Design.md                   # Design decisions and trade-offs
├── PRD.md                      # Project requirements
├── Phases.md                   # Development phases / status
├── Rules.md                    # Development standards
│
├── src/
│   ├── data_loader.py          # NASA PCoE parsing + feature extraction
│   ├── ecm_thevenin.py         # 1-RC ECM / EKF utilities
│   └── models/
│       ├── pi_mdcnet.py        # MonARCH / SLAC architecture
│       ├── baselines.py        # Baseline models
│       └── intracycle_soc.py   # Intra-cycle SoC GRU
│
├── experiments/
│   ├── train.py                # LOBO training / benchmarking
│   ├── ablation.py             # Component ablations
│   ├── rul_validation.py       # Causal RUL validation
│   ├── eval_rul_cell_breakdown.py
│   ├── eval_naive_soc.py
│   └── run_intracycle_soc.py
│
├── embedded/
│   ├── export_onnx.py          # ONNX export + verification
│   └── esp32/                  # C99 ESP32-S3 streaming engine
│
├── cloud/                      # Serverless fleet analytics pipeline
├── dashboard/                  # Web monitoring dashboard
├── artifacts/                  # Models, metrics, figures and reports
├── scripts/                    # Dataset utilities
└── tests/                      # Unit / integration tests
```

## Quick Start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify the dataset

```bash
python scripts/download_data.py
```

### 3. Run tests

```bash
pytest tests/ -v
```

### 4. Run experiments

```bash
python experiments/train.py
python experiments/ablation.py
python experiments/rul_validation.py
python experiments/run_intracycle_soc.py
```

### 5. Export the edge model

```bash
python embedded/export_onnx.py
```

## Edge Deployment

The embedded path is intentionally separate from the training stack:

```text
PyTorch model
     │
     ▼
ONNX export / verification
     │
     ▼
~318 KB edge model
     │
     ▼
ESP32-S3 / Xtensa LX7
     │
     ▼
C99 streaming inference
```

The embedded implementation is designed for constrained deployment rather than assuming a desktop Python runtime exists on the device.

## Cloud Fleet Architecture

The repository also contains a serverless fleet-analytics design for simulated battery telemetry:

**AWS IoT Core → Lambda → DynamoDB/S3 → Step Functions → ONNX inference → API Gateway → dashboard**, with CloudWatch/SNS-based monitoring and alerting.

See [`Architecture.md`](Architecture.md) for the complete cloud and model architecture.

## Technology

**Python · PyTorch · NumPy · SciPy · Pandas · Scikit-learn · XGBoost · Optuna · ONNX Runtime · C99 · ESP32-S3 · AWS IoT Core · AWS Lambda · DynamoDB · S3 · Step Functions · API Gateway · Plotly**

## Documentation

- [`Architecture.md`](Architecture.md) — mathematical and system architecture
- [`Design.md`](Design.md) — design rationale and trade-offs
- [`PRD.md`](PRD.md) — requirements
- [`Phases.md`](Phases.md) — implementation phases
- [`artifacts/forensic_verification_report.md`](artifacts/forensic_verification_report.md) — claim-by-claim verification

## License

MIT License.
