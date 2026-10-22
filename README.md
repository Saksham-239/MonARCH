# MonARCH — Physics-Guided Battery State & RUL Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Embedded: ESP32--S3](https://img.shields.io/badge/Embedded-ESP32--S3%20%2F%20Xtensa-red.svg)](embedded/esp32/)
[![Edge: ONNX Runtime](https://img.shields.io/badge/Edge-ONNX%20Runtime-blue.svg)](artifacts/)
[![Tests: Pytest](https://img.shields.io/badge/Tests-Passing-success.svg)](tests/)

**MonARCH** (*Monotonic Aging and Recovery-Constrained Neural State-Space Network*) is an open-source, physics-guided deep-learning framework for joint estimation of **State of Charge (SoC), State of Health (SoH), and Remaining Useful Life (RUL)** in lithium-ion battery management systems (BMS).

Standard data-driven models frequently predict physically implausible battery behaviors—such as spontaneous capacity recovery during continuous discharge, non-causal label leakage, or non-monotonic degradation paths. MonARCH embeds governing physical principles directly into the neural network architecture rather than relying on weak loss regularization.

> **MonARCH = Architecturally Monotonic Degradation + Hard-Gated Reversible Recovery + Two-Timescale Decoupling + Causal RUL Extrapolation + Ultra-Low-Footprint Edge C99 Deployment.**

---

## System Overview & Timescale Decoupling

Battery telemetry operates across fundamentally different physical timescales: fast electrochemical responses within a single discharge cycle (seconds to minutes) and slow macroscopic capacity fade across aging cycles (weeks to years). MonARCH explicitly decouples these dynamics into a coordinated two-timescale architecture:

```text
                            Lithium-Ion Telemetry [V(t), I(t), T(t)]
                                               │
                   ┌───────────────────────────┴───────────────────────────┐
                   ▼                                                       ▼
        Micro-Timescale: Intra-Cycle                            Macro-Timescale: Inter-Cycle
        Continuous State of Charge (SoC)                        Degradation & Recovery (SLAC Core)
                   │                                                       │
        Lightweight Sequence GRU                                ┌──────────┴──────────┐
        [V(t), I(t), t]                                         ▼                     ▼
                   │                                     Irreversible          Reversible
                   ▼                                     Damage D(t)           Recovery R(t)
        Coulomb-Normalized SoC(t)                        (Strictly             (Hard-Gated
                                                          Monotonic)            by Rest)
                                                                │                     │
                                                                └──────────┬──────────┘
                                                                           ▼
                                                                  SoH = SoH₀ − D(t) + R(t)
                                                                           │
                                                                           ▼
                                                                Causal RUL Extrapolation
                                                                (Latent D(t) → Fail Threshold)
```

---

## Core Inductive Physics

### 1. Architecturally Monotonic Damage (D_t)
Capacity fade caused by irreversible active material loss, solid-electrolyte interphase (SEI) layer growth, and lithium plating cannot spontaneously reverse. MonARCH structurally parameterizes the damage increment using a positive-definite activation:

```text
ΔD_t = λ * Softplus(f_θ(s_t))
D_t  = D_(t-1) + ΔD_t
```

where `λ = 0.003`, `s_t = [T_mean, C-rate]` is the operational stressor vector, and `Softplus(u) = ln(1 + e^u) > 0`. Because the Softplus increment is strictly non-negative (`ΔD_t >= 0`) by construction, the cumulative damage trajectory cannot spontaneously decrease, ensuring physical degradation monotonicity across battery life.

### 2. Hard-Gated Reversible Recovery (R_t)
Battery relaxation after qualifying rest intervals can produce apparent capacity recovery. MonARCH models this separately using a physical gating mechanism:

```text
R_t = rest_flag_t * R_head(rest_features, h_t)
```

where `rest_flag_t ∈ {0, 1}`. During continuous cycling with no rest period (`rest_flag_t == 0`), the recovery branch is structurally forced to zero (`R_t == 0`), preventing the network from hallucinating unphysical capacity jumps during active cycling.

### 3. Causal Remaining Useful Life (RUL) Extrapolation
Rather than training a regressor that observes the future or relies on arbitrary trajectory smoothing, MonARCH extrapolates the monotonic latent damage trajectory forward to the configured end-of-life threshold:

```text
D_fail = SoH_0 - (1 - EOL_FADE_FRACTION)
```

where `EOL_FADE_FRACTION = 0.20` (configured in `src/data_loader.py` and `experiments/protocol.py`) corresponds to the standard 80% SoH end-of-life threshold (`EOL capacity = 1.60 Ah` for rated 2.0 Ah NASA cells). This standard definition ensures valid, non-censored ground truth RUL across all benchmark cells (including cell B0007, whose minimum capacity is 1.4005 Ah). Using a linear projection on the causal rolling damage history (`k = 10`), MonARCH projects cycles remaining until `D_t >= D_fail` with zero future-cycle leakage.

### 4. Ultra-Compact Edge Deployment (ESP32-S3 / C99)
- **Persistent State:** Exactly **52 bytes** (`sizeof(pimdcnet_state_t)`: 4B `D_(t-1)` + 40B rolling `D` buffer + 4B cycle counter + 4B calibrated `SoH_0`).
- **Compute Complexity:** 21,061 float32 parameters (82.27 KB Flash ROM) and 23,504 MACs per cycle step.
- **Microcontroller Safety:** Peak stack frame of **4,960 bytes** (~4.84 KB, verified via GCC `-fstack-usage`), fitting comfortably into ESP32-S3 SRAM with zero heap allocation.
- **Numerical Parity:** Exact bitwise match for SoH and `D_t` (0.000 difference) and `< 1e-7` maximum absolute difference for `R_t`, SoC, and RUL against PyTorch.
- **Edge ONNX Graph:** 133,435 bytes (~133 KB FP32) and 124,950 bytes (~125 KB INT8 quantized).

---

## Empirical Benchmark & Verification Results

### Evaluation Scope
Aligned with [`Architecture.md`](Architecture.md), the **Primary Benchmark Suite** is evaluated under a strict **Leave-One-Battery-Out (LOBO)** cross-validation protocol across the 4 **NASA PCoE** 24°C battery cells (`B0005`, `B0006`, `B0007`, `B0018`) over 5 random seeds (`[42, 100, 2024, 2025, 2026]`). To assess cross-dataset and cross-chemistry generalization, the evaluation scope is extended to:
1. **NASA PCoE (Primary):** 4 cylindrical 18650 cells run-to-failure at 24°C.
2. **CALCE (Generalization):** 4 prismatic LCO cells (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`) with 800+ extended cycles.
3. **Oxford (Dynamic Telemetry):** 4 Kokam pouch cells (`Cell1`, `Cell2`, `Cell3`, `Cell4`) subjected to dynamic urban Artemis drive cycles.

### 1. State of Health (SoH) Benchmark Across 12 Folds (Mean ± Std)

| Model Category | Evaluation Paradigm | NASA SoH RMSE | CALCE SoH RMSE | Oxford SoH RMSE | Overall 12-Fold RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Persistence (SoH[t-1])** | 1-step lag baseline | 0.0090 ± 0.0030 | 0.0028 ± 0.0001 | 0.0048 ± 0.0001 | **0.0055 ± 0.0031** |
| **Persistence + Drift** | Causal linear drift | 0.0089 ± 0.0030 | 0.0028 ± 0.0001 | 0.0041 ± 0.0003 | **0.0052 ± 0.0031** |
| **Random Forest + lag** | Supervised lag regressor | 0.0198 ± 0.0084 | 0.0005 ± 0.0008 | 0.0008 ± 0.0001 | 0.0071 ± 0.0104 |
| **XGBoost + lag** | Gradient boosted trees | 0.0237 ± 0.0061 | 0.0008 ± 0.0008 | 0.0013 ± 0.0002 | 0.0086 ± 0.0116 |
| **MonARCH / PI-MDCNet** | Lag-conditioned | 0.0109 ± 0.0055 | 0.0045 ± 0.0004 | 0.0031 ± 0.0003 | **0.0062 ± 0.0045** |
| **MonARCH / PI-MDCNet (C2)** | **Strict Open-Loop (Zero Test Labels)** | **0.0804 ± 0.0582** | **0.0173 ± 0.0011** | **0.0133 ± 0.0009** | **0.0370 ± 0.0442** |

*Note on Evaluation Rigor:* As documented in [`artifacts/v5/REPORT.md`](artifacts/v5/REPORT.md), models conditioned on test-cell `SoH[t-1]` act as one-step filters. Under open-loop evaluation where no test-cell ground truth is ever provided, MonARCH tracks the full multi-hundred cycle capacity fade trajectory with an overall RMSE of 0.0370.

### 2. NASA Remaining Useful Life (RUL) Prediction

Evaluated across 4 NASA test folds × 5 seeds across 4 prognostic extrapolation methods and 2 evaluation modes (160 evaluation runs total; 20 paired runs per method) with uniform cycle normalization (`t / 1000`):

| Evaluation Mode | Prognostic Extrapolation Method | Mean MAE (Cycles) | Mean RMSE (Cycles) | B0005 MAE | B0006 MAE | B0007 MAE | B0018 MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Open-Loop (C2)** | **Latent D_t Extrapolation (MonARCH)** | **7.02 ± 3.93** | **12.00 ± 6.65** | 8.09 | 9.94 | 8.84 | 1.23 |
| | Neural RUL Head | 7.10 ± 5.27 | 12.43 ± 8.44 | 8.47 | 4.74 | 13.73 | 1.46 |
| | Statistical Degradation Decomposition | 23.93 ± 22.86 | 41.65 ± 38.48 | 35.18 | 5.41 | 50.71 | 4.40 |
| | Raw SoH Extrapolation | 24.26 ± 25.06 | 49.88 ± 51.39 | 42.94 | 4.19 | 48.73 | 1.16 |
| **Lag-Conditioned** | Neural RUL Head | 6.88 ± 3.92 | 11.83 ± 6.59 | 8.32 | 4.95 | 11.60 | 2.63 |
| | Latent D_t Extrapolation | 9.44 ± 5.42 | 16.78 ± 7.85 | 12.03 | 5.16 | 15.84 | 4.73 |

MonARCH's Latent `D_t` extrapolation achieves a **3.4x reduction in RUL error** compared to standard statistical and raw-capacity extrapolation baselines (~7.0 cycles vs ~24.0 cycles MAE).

### 3. Open-Loop Ablation Study (NASA Folds × 5 Seeds)

| Architectural Variant | SoH RMSE | SoH MAE | RUL MAE (Cycles) | Monotonic dD/dt >= 0 | Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full SLAC (MonARCH)** | 0.0804 ± 0.0582 | 0.0677 ± 0.0455 | **8.84 ± 2.57** | **Verified True** | Baseline reference |
| **No Cross-Attention** | 0.0803 ± 0.0582 | 0.0676 ± 0.0454 | 8.85 ± 2.59 | **Verified True** | p = 0.9274 (neutral impact) |
| **No Hard Gating** | 0.0503 ± 0.0378 | 0.0427 ± 0.0311 | 7.38 ± 1.44 | **Verified True** | Soft recovery provides lower fitting error |
| **No Monotonicity** | 0.0624 ± 0.0207 | 0.0506 ± 0.0225 | **75.31 ± 47.21** | **Violated (< 0)** | **RUL error explodes 8.5x** |

**Crucial Insight:** While unconstrained neural networks can slightly reduce one-step fitting loss by oscillating, removing architectural monotonicity destroys prognostic generalization—causing RUL MAE to explode from **8.84 to 75.31 cycles**. Inductive monotonicity is vital for long-term health forecasting.

### 4. Continuous Intra-Cycle State of Charge (SoC) Tracking

Continuous evaluation across operational discharge curves per battery aging tier:

| Dataset | Aging Tier | Naive Static CC | Intra-Cycle GRU | SOH-Conditioned CC | Model-Free Lag CC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NASA** | Fresh (>= 90%) | 0.0425 ± 0.0094 | 0.0200 ± 0.0059 | 0.0074 ± 0.0022 | **0.0067 ± 0.0036** |
| | Mid-Life (80-90%) | 0.0887 ± 0.0032 | 0.0163 ± 0.0075 | 0.0074 ± 0.0035 | **0.0068 ± 0.0027** |
| | Near-EOL (< 80%) | 0.1774 ± 0.0250 | 0.0332 ± 0.0254 | 0.0084 ± 0.0030 | **0.0072 ± 0.0022** |
| | **Overall** | 0.1427 ± 0.0245 | 0.0293 ± 0.0200 | 0.0083 ± 0.0025 | **0.0073 ± 0.0018** |
| **Oxford** | Fresh (>= 90%) | 0.0375 ± 0.0009 | 0.0209 ± 0.0001 | **0.0028 ± 0.0003** | 0.0040 ± 0.0002 |
| | Mid-Life (80-90%) | 0.0907 ± 0.0005 | 0.0207 ± 0.0001 | **0.0029 ± 0.0002** | 0.0035 ± 0.0005 |
| | Near-EOL (< 80%) | 0.1336 ± 0.0007 | 0.0213 ± 0.0001 | **0.0026 ± 0.0006** | 0.0039 ± 0.0004 |
| | **Overall** | 0.0917 ± 0.0012 | 0.0209 ± 0.0001 | **0.0028 ± 0.0001** | 0.0037 ± 0.0001 |

Conditioning Coulomb counting on MonARCH's predicted SoH cuts near-EOL tracking error from **17.7% down to < 0.9%** by dynamically adjusting active cell capacity `Q_aged = SoH_t * Q_nom`.

---

## Repository Structure

```text
.
├── Architecture.md             # Complete system & cloud technical specification
├── Design.md                   # Industrial UI design tokens & visual system
├── PRD.md                      # Detailed project requirements document
├── Rules.md                    # Engineering principles & anti-leakage standards
├── .clangd                     # Clangd LSP configuration for embedded C toolchain
├── compile_commands.json       # Compilation database for C99 embedded profiler
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
│
├── src/                        # Core Python Modules
│   ├── data_loader.py          # Multi-dataset parser (NASA, CALCE, Oxford) & ECM extraction
│   ├── ecm_thevenin.py         # 1-RC equivalent circuit & online parameter identification
│   └── models/
│       ├── pi_mdcnet.py        # MonARCH / SLAC neural network architecture
│       ├── baselines.py        # Coulomb Counting, EKF, RF, XGBoost, LSTM baselines
│       └── intracycle_soc.py   # Auxiliary intra-cycle continuous SoC GRU
│
├── experiments/                # Verification & Training Pipelines
│   ├── protocol.py             # Centralized deterministic seeds, metrics & LOBO folds
│   ├── train.py                # Unified training & evaluation with open-loop/recursive modes
│   ├── ablation.py             # 4-way architectural ablation runner
│   ├── rul_validation.py       # Multi-head RUL extrapolation validation
│   ├── eval_rul_cell_breakdown.py # Per-cell RUL diagnostic breakdown
│   ├── eval_naive_soc.py       # Zero-learning SoC comparator
│   └── run_intracycle_soc.py   # High-frequency intra-cycle SoC evaluation
│
├── embedded/                   # Edge C99 Deployment & ONNX Export
│   ├── export_onnx.py          # PyTorch → ONNX FP32 & INT8 quantization + verification
│   └── esp32/                  # Pure C99 BMS Streaming Engine
│       ├── CMakeLists.txt      # ESP-IDF / Native CMake build configuration
│       └── main/
│           ├── pi_mdcnet_inference.h # 52-byte persistent state struct & API
│           ├── pi_mdcnet_inference.c # C99 inference forward pass implementation
│           ├── pi_mdcnet_weights.h   # Static weights header (21,061 float32 constants)
│           └── main.c                # Xtensa LX7 240 MHz cycle profiler & harness
│
├── scripts/                    # Automated Benchmark & Verification Scripts
│   ├── download_data.py        # Automated multi-dataset downloader & cache verifier
│   ├── run_v5_soc.py           # Continuous intra-cycle SoC benchmark runner
│   ├── run_v5_rul.py           # RUL benchmark & per-cell trajectory figure generator
│   ├── run_v5_c4_ablation.py   # Strict open-loop ablation verification
│   └── run_v5_part_b_c.py      # Main 12-fold SoH benchmark runner
│
├── artifacts/                  # Benchmark Artifacts & Exported Models
│   ├── pi_mdcnet_edge.onnx     # Streaming FP32 ONNX model (~133 KB)
│   ├── pi_mdcnet_edge_int8.onnx# Quantized INT8 ONNX model (~125 KB)
│   ├── forensic_verification_report.md # Comprehensive claim-by-claim forensic audit
│   └── v5/                     # Latest Revision Benchmark Results
│       ├── REPORT.md           # Full empirical verification report (Parts A-F)
│       ├── soh_benchmark_summary.csv # 12-fold SoH summary across all models
│       ├── rul_benchmark_summary.csv # Multi-mode RUL summary
│       ├── soc_comparator_summary.csv# Intra-cycle SoC comparator summary
│       ├── ablation_openloop_summary.csv # NASA open-loop ablation summary
│       └── figures/            # High-resolution per-cell RUL vector curves (PDF/PNG)
│
├── cloud/                      # AWS Serverless Fleet Analytics Pipeline
│   └── lambdas/                # Ingest, feature extraction, ONNX inference & API handlers
├── dashboard/                  # Industrial dark-mode real-time telemetry dashboard
└── tests/                      # Automated Test Suite (17 unit & integration tests)
```

---

## Quick Start Guide

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Saksham-239/MonARCH.git
cd MonARCH

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Verify Datasets

Automatically download, unpack, and verify checksums for NASA PCoE, CALCE, and Oxford battery datasets:

```bash
python scripts/download_data.py
```

### 3. Run Automated Tests

Execute the unit and integration test suite:

```bash
pytest tests/ -v
```

The test suite validates:
- Architectural forward pass shapes and gradient flow.
- Parameter-level non-negative damage monotonicity ($\Delta D_t \ge 0$).
- Exact zero recovery gating during active cycling ($R_t \equiv 0$ when `rest_flag == 0`).
- Strict test-label isolation under open-loop and recursive modes.
- Numerical equivalence between the pure C99 engine, ONNX runtime, and PyTorch.

### 4. Run Benchmark Suite

```bash
# 1. Main SoH 12-fold benchmark across all models
python scripts/run_v5_part_b_c.py

# 2. Open-loop component ablation study
python scripts/run_v5_c4_ablation.py

# 3. RUL benchmark and per-cell trajectory figure generation
python scripts/run_v5_rul.py

# 4. Continuous intra-cycle SoC benchmark
python scripts/run_v5_soc.py
```

### 5. Build and Run the C99 Embedded Profiler

Compile and run the standalone C99 streaming inference engine locally using any C99 compiler:

```bash
# Compile host profiler
gcc -O3 -fno-strict-aliasing embedded/esp32/main/main.c embedded/esp32/main/pi_mdcnet_inference.c -Iembedded/esp32/main -lm -o embedded/esp32_demo

# Run the profiler
./embedded/esp32_demo
```

For ESP32-S3 deployment with ESP-IDF:
```bash
cd embedded/esp32
idf.py set-target esp32s3
idf.py build flash monitor
```

---

## Serverless Cloud Fleet Architecture

MonARCH includes an event-driven AWS serverless pipeline designed for zero idle cost ($0.00/month when no telemetry arrives):

```text
Battery Telemetry (MQTT)
         │
         ▼
    AWS IoT Core  ──(SQL Rule)──►  AWS Lambda (Ingestion)
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             Amazon DynamoDB (Hot State)         Amazon S3 (Cold Telemetry)
                        │                                   │
                        └─────────────────┬─────────────────┘
                                          ▼
                               AWS Step Functions Orchestrator
                                          │
                               ┌──────────┴──────────┐
                               ▼                     ▼
                     Feature Extraction       ONNX Runtime Lambda
                     (1-RC ECM & dQ/dV)       (pi_mdcnet_edge.onnx)
                                                     │
                               ┌─────────────────────┴─────────────────────┐
                               ▼                                           ▼
                      Amazon API Gateway                          Amazon SNS
                      (REST Fleet Endpoints)                      (Thermal & EOL Alerts)
                               │
                               ▼
                   Responsive Web Dashboard
```

Complete configuration templates, CloudWatch alarm rules, and IAM policies are detailed in [`Architecture.md`](Architecture.md).

---

## Documentation

- [`Architecture.md`](Architecture.md) — Mathematical formulations, equivalent circuit models, and cloud infrastructure.
- [`artifacts/v5/REPORT.md`](artifacts/v5/REPORT.md) — Comprehensive v5 benchmark verification report and empirical findings.

---

## License

This project is licensed under the [MIT License](LICENSE).
