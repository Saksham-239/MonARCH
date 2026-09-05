# MonARCH: A Monotonic Aging and Recovery-Constrained Neural State-Space Network for Battery RUL Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Edge%20Optimized-005ced.svg)](https://onnxruntime.ai/)
[![Target](https://img.shields.io/badge/Embedded-ESP32--S3%20Xtensa%20LX7-brightgreen.svg)](embedded/esp32/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **MonARCH** (*Monotonic Aging and Recovery-Constrained Neural State-Space Network*, formerly *PI-MDCNet v4*) is a physics-guided deep learning framework for co-estimating State of Charge (SoC), State of Health (SoH), and Remaining Useful Life (RUL) across coupled electrochemical timescales in Lithium-ion batteries.

---

## Key Highlights

- **Decoupled Two-Timescale Architecture**:
  - **Micro-Timescale (Intra-Cycle)**: Real-time continuous $\text{SoC}(t)$ estimation using a lightweight gated sequence network trained against a retrospective cycle-normalized reference (strictly withholding cycle capacity $Q_{\text{cycle}}$ at inference).
  - **Macro-Timescale (Inter-Cycle)**: Per-cycle degradation tracking using the **Split-Latent Aging Core (SLAC)**.
- **Architectural Monotonicity**:
  $$\text{SoH}_t = \text{SoH}_0 - D_t + R_t$$
  $$\Delta D_t = \lambda \cdot \text{Softplus}(f_\theta(s_t)) > 0, \quad D_t = D_{t-1} + \Delta D_t$$
  Cumulative damage $D_t$ is strictly monotonic by parametric construction ($\lambda = 0.003 > 0$, $\text{Softplus}(u) > 0$), preventing unphysical health rebounds without relying on fragile soft penalty weights.
- **Hard-Gated Rest-Induced Recovery**:
  $$R_t = \text{rest\_flag}_t \cdot R_{\text{head}}(\text{rest\_features}, h_t)$$
  Prevents false capacity regeneration during continuous cycling while capturing legitimate electrochemical relaxation following extended rest intervals.
- **Causal RUL Extrapolation**:
  Forecasts remaining operating cycles to End of Life (EOL, 70% rated capacity = 1.40 Ah) by projecting the monotonic latent damage trajectory $D_t \to D_{\text{fail}}$ with explicit treatment of right-censored trajectories (e.g., NASA B0007).
- **Zero-Heap C99 Embedded Engine**:
  Includes an ultra-compact C99 analytical streaming engine designed for the **ESP32-S3 (Xtensa LX7)** with a **32-byte persistent RAM state footprint** (`pimdcnet_state_t`), alongside an exported ~318 KB ONNX edge runtime graph.

---

## System Architecture

```
+--------------------------------------------------------------------------------------------------+
|                                      MonARCH ENGINE                                              |
|                                                                                                  |
|   NASA PCoE Datasets (B0005: 2.7V, B0006: 2.5V, B0007: 2.2V, B0018: 2.5V cutoff at 24°C)         |
|             |                                                                                    |
|             v                                                                                    |
|   [src/data_loader.py] ---> 1-RC Thevenin Parameter Estimation + dQ/dV Spectra + Rest Flags      |
|             |                                                                                    |
|             v                                                                                    |
|   [Leave-One-Battery-Out CV] ---> Train on 3 cells, evaluate on 1 held-out cell                  |
|             |                                                                                    |
|             +---> Baselines (Coulomb Counting, EKF, Random Forest, XGBoost, Vanilla LSTM)        |
|             +---> MonARCH (SLAC: Parametric Monotonic D_t + Hard-Gated R_t + Cross-Attention)   |
|             +---> 4-Way Component Ablation (Full, NoCrossAttn, NoHardGating, NoMono)             |
|             |                                                                                    |
|             v                                                                                    |
|   [artifacts/pi_mdcnet_edge.onnx] ----> Edge-optimized model graph (~318 KB, OpSet 14)           |
|   [embedded/esp32] --------------> C99 Streaming Engine (Xtensa LX7, 32-Byte State RAM)          |
+--------------------------------------------------------------------------------------------------+
```

---

## Directory Structure

```
├── Architecture.md         # Full system architecture and mathematical specification
├── Design.md               # Design documentation and trade-off analysis
├── PRD.md                  # Project Requirements Document
├── Rules.md                # Development and coding standards
├── Phases.md               # Project development phases and status
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
│
├── src/                    # Core framework modules
│   ├── data_loader.py      # NASA PCoE parser, Thevenin 1-RC extraction, feature pipeline
│   └── models/
│       ├── pi_mdcnet.py    # MonARCH / PI-MDCNet model and SLAC core
│       ├── baselines.py    # Standard ML & deep learning baselines (RF, XGB, LSTM, EKF)
│       └── intracycle_soc.py # Micro-timescale SoC estimation module
│
├── experiments/            # Experiment and evaluation drivers
│   ├── train.py            # Training pipeline with LOBO-CV
│   ├── evaluate.py         # SoH benchmarking and evaluation metrics
│   ├── ablation.py         # 4-way ablation suite (Full, NoCrossAttn, NoHardGate, NoMono)
│   ├── rul_validation.py   # RUL extrapolation and right-censoring validation
│   ├── run_intracycle_soc.py # Intra-cycle SoC benchmark
│   └── protocol.py         # Evaluation protocol definitions
│
├── embedded/               # Edge deployment and microcontroller firmware
│   ├── export_onnx.py      # ONNX export and runtime verification script
│   └── esp32/              # C99 firmware for ESP32-S3 (Xtensa LX7)
│       ├── CMakeLists.txt
│       └── main/
│           ├── main.c
│           └── pi_mdcnet_inference.h # Zero-heap streaming inference header
│
├── data/raw/               # NASA PCoE battery aging mat files (B0005, B0006, B0007, B0018)
│
├── artifacts/              # Generated figures, benchmark results, and verification report
│   ├── figures/            # Trajectory and decomposition plots
│   ├── benchmark_results.csv
│   ├── ablation_table.csv
│   ├── rul_validation_results.csv
│   ├── forensic_verification_report.md # Comprehensive 22-claim audit report
│   └── pi_mdcnet_edge.onnx # Exported edge model graph
│
├── scripts/
│   └── download_data.py    # Automated dataset download and extraction script
│
└── tests/                  # Unit and integration test suite
    ├── test_data_loader.py
    ├── test_pi_mdcnet.py
    ├── test_baselines.py
    └── test_evaluate.py
```

---

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Saksham-239/MonARCH.git
cd MonARCH

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Verification

The repository includes the NASA Ames Prognostics Center of Excellence (PCoE) battery aging datasets in `data/raw/`. To verify or redownload them:

```bash
python scripts/download_data.py
```

### 3. Running Unit Tests

```bash
pytest tests/ -v
```

### 4. Training & LOBO-CV Evaluation

```bash
# Train and benchmark MonARCH against baselines
python experiments/train.py

# Run SoH model evaluation
python experiments/evaluate.py

# Run RUL causal extrapolation
python experiments/rul_validation.py

# Run 4-way architectural ablation study
python experiments/ablation.py

# Run micro-timescale intra-cycle SoC evaluation
python experiments/run_intracycle_soc.py
```

### 5. Exporting Edge Artifacts

```bash
python embedded/export_onnx.py
```

---

## Forensic Verification & Defensibility

Every claim in this repository has been verified and grounded against raw data and empirical code. See [`artifacts/forensic_verification_report.md`](artifacts/forensic_verification_report.md) for the complete 22-claim audit table. Key highlights:
- **Verified Cell Cutoffs**: NASA PCoE protocols verified from raw sensor data: B0005 (2.7 V), B0006 (2.5 V), B0007 (2.2 V), B0018 (2.5 V).
- **Monotonicity Defensibility**: Monotonicity of the damage component $D_t$ is enforced by parametric formulation ($\Delta D_t = \lambda \cdot \text{Softplus} > 0$), empirically verified with zero violations across all test runs.
- **Statistical Honesty**: With $N=4$ battery trajectories ($N=3$ reaching EOL), inferential tests are reported as exploratory.

---

## License

This project is licensed under the MIT License.
