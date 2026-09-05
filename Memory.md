# Project Memory & Advancements Log: PI-MDCNet

**Project:** Physics-Informed Multi-Domain Convolutional Network (PI-MDCNet v4 / SLAC)  
**Workspace:** `c:\Users\saksh\OneDrive\Documents\AWS ML`  
**Last Updated:** September 2026 (Post-Forensic Audit Canonical Run)  
**Purpose:** Maintain complete project context, technical memory, architectural decisions, chronological milestones, and empirical results across AI sessions and development tools.

---

## 1. Project Overview & Core Philosophy

PI-MDCNet addresses lithium-ion battery prognostics (co-estimation of State of Charge [SoC], State of Health [SoH], and Remaining Useful Life [RUL]) across multi-timescale operational dynamics.

### Architectural Identity: Split-Latent Aging Core (SLAC)
Instead of treating battery aging as an unconstrained regression problem, SLAC embeds electrochemical inductive biases directly into the network architecture:
1. **Monotonic Irreversible Cumulative Damage ($D_t$):**
   $$\Delta D_t = \lambda \cdot \text{Softplus}(f_\theta(s_t)) > 0, \quad D_t = D_{t-1} + \Delta D_t \quad (\lambda = 0.003)$$
   Guarantees non-decreasing damage by parametric construction without soft penalty losses.
2. **Decoupled Latent State of Health:**
   $$\text{SoH}_t = \text{SoH}_0 - D_t + R_t$$
   Explicitly isolates irreversible degradation $D_t$ from reversible electrochemical relaxation $R_t$.
3. **Hard-Gated Rest Relaxation ($R_t$):**
   $$R_t = \text{rest\_flag}_t \cdot R_{\text{head}}$$
   Rest recovery is strictly $0.0$ on active cycling; relaxation occurs only after verified rest intervals.
4. **Causal Damage-Latent RUL Extrapolation:**
   Extrapolates RUL from the monotonic latent state $D_t$ using only historical cycle observations ($[t-k+1, t]$), preventing future target leakage.
5. **Micro-Scale Intra-Cycle SoC Tracking:**
   A lightweight GRU estimates real-time SoC from high-frequency $[V(t), I(t), t]$ telemetry during discharge.

---

## 2. Chronological Milestones & Implementation History

### Phase 0: Project Initiation & Specifications
* **Foundational Specs Created:** [`PRD.md`](file:///c:/Users/saksh/OneDrive/Documents/AWS%20ML/PRD.md), [`Architecture.md`](file:///c:/Users/saksh/OneDrive/Documents/AWS%20ML/Architecture.md), [`Rules.md`](file:///c:/Users/saksh/OneDrive/Documents/AWS%20ML/Rules.md), [`Phases.md`](file:///c:/Users/saksh/OneDrive/Documents/AWS%20ML/Phases.md), [`Design.md`](file:///c:/Users/saksh/OneDrive/Documents/AWS%20ML/Design.md).
* **Cloud Cost Boundary:** Strict $0.00 idle cost limit; prohibition of hourly AWS services (no Kinesis, SageMaker endpoints, or Glue).

### Phase 1: Data Ingestion & NASA PCoE Protocol Corrections
* **Raw Datasets Processed:** NASA PCoE battery cells `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`.
* **Discharge Cutoff Correction:** Audited raw voltage arrays against official NASA documentation and corrected the historical documentation error:
  - B0005: **2.7 V**
  - B0006: **2.5 V**
  - B0007: **2.2 V** *(corrected from 2.7 V)*
  - B0018: **2.5 V**
* **Feature Engineering:** Extracted $dQ/dV$ differential capacity curves (30 voltage bins), fitted 1RC Thevenin equivalent circuit parameters ($R_0, R_1, C_1$), computed cycle stressor vector $s_t$, and calculated rest durations.

### Phase 2: Fairly-Tuned Baselines Engine
* **Implemented Baselines:**
  - Zero-learning nominal Coulomb Counting ($Q_{\text{rated}} = 2.0\text{ Ah}$)
  - 1RC Extended Kalman Filter (`EKF_1RC`)
  - Random Forest Regressor (`RandomForest`)
  - XGBoost Regressor (`XGBoost`)
  - Vanilla LSTM Recurrent Network (`VanillaLSTM`)
  - Statistical Trend-Seasonal Decomposition (`StatDecomp`)
* **Validation Rigor:** Leave-One-Battery-Out Cross-Validation (LOBO-CV) with strict train/test isolation and zero test data leakage in feature scalers.

### Phase 3: PI-MDCNet v4 (SLAC) Implementation
* Built multi-head convolutional/attention encoders for electrical ($dQ/dV$, ECM) and thermal features.
* Implemented the cross-attention fusion layer between electrical and thermal modalities.
* Parameterized the positive damage generator using $\lambda \cdot \text{Softplus}(\cdot)$.
* Implemented multi-task loss function balancing SoH MSE, SoC MSE, and RUL supervision.

### Phase 4: Full Benchmark Execution & Canonical Run
* Conducted full 4-fold LOBO cross-validation across 3 random seeds (42, 43, 44), generating 12 experimental runs per model.
* Archived all legacy intermediate snapshots into `artifacts_archive/` and executed a single fresh canonical run across the entire pipeline.

### Phase 5: 4-Way Component Ablation Study
* **Ablation Variants:**
  1. `Full_SLAC`: Complete architecture.
  2. `Ablation_NoCrossAttn`: Replaces cross-attention fusion with feature concatenation.
  3. `Ablation_NoHardGating`: Removes binary rest flag gating on $R_t$.
  4. `Ablation_NoMonotonicity`: Removes parametric Softplus constraint on $\Delta D_t$.
* **Key Finding:** Monotonicity is the decisive inductive mechanism ($+203.5\%$ error explosion when removed); cross-attention provides negligible isolated gain ($\Delta = 0.0001$).

### Phase 6: Embedded Deployment & Edge Verification
* **Target Hardware:** ESP32-S3 dual-core 32-bit **Xtensa LX7** microcontroller (corrected from ARM Cortex-M).
* **Streaming C99 Engine:** Implemented in `embedded/esp32/main/pi_mdcnet_inference.c` with zero heap allocation and exactly **32 bytes** of persistent BMS state (`pimdcnet_state_t`).
* **ONNX Compilation:** Exported `artifacts/pi_mdcnet_edge.onnx` (318,034 bytes, 67,013 float32 parameters).
* **Sequential Parity:** Verified that ONNX Runtime matches PyTorch recurrent outputs within $< 1.19 \times 10^{-7}$ across 50 consecutive cycles.

### Phase 7: Forensic Audit Pass & Publication Defensibility
* Full academic honesty audit addressing an adversarial review panel:
  - Eliminated all "first ever" claims (acknowledged prior literature: Qin et al. 2016 for rest recovery, monotonic PINNs for degradation constraints).
  - Clarified statistical limits: formal significance ($p < 0.05$) is mathematically impossible on $N=4$ ($p \ge 0.125$) and $N=3$ ($p \ge 0.250$).
  - Clarified SoC reference: retrospective cycle-normalized reference ($1 - q(t)/Q_{\text{cycle}}$), not independent sensor ground truth.
  - Documented B0007 right-censoring at 1.40 Ah EOL threshold.
  - Formulated 22-claim verification table, 11-question faculty oral defense script, and publication-safe contribution statement.

---

## 3. Canonical Benchmark Numbers (Off-Disk Source of Truth)

All metrics below trace directly to the single fresh canonical rerun in `artifacts/`:

### 3.1 Macro SoH Tracking (`artifacts/comparison_table.csv`)
* **PI-MDCNet (Full SLAC):**
  - SoH RMSE: **0.0573 ± 0.0138** (pooled over 12 runs: 4 folds × 3 seeds)
  - Across-fold cell dispersion: $\sigma_{\text{cell}} = \mathbf{0.0099}$ ($N=4$ fold means: B0005=0.0493, B0006=0.0715, B0007=0.0559, B0018=0.0522)
  - Mean within-cell seed dispersion: $\bar{\sigma}_{\text{seed}} = \mathbf{0.0088}$
  - SoH MAE: **0.0459 ± 0.0110**
* **Baselines (SoH RMSE):**
  - `RandomForest`: $0.0300 \pm 0.0066$
  - `XGBoost`: $0.0306 \pm 0.0076$
  - `VanillaLSTM`: $0.0444 \pm 0.0132$
  - `StatDecomp`: $0.1091 \pm 0.0112$

### 3.2 Component Ablation Results (`artifacts/ablation_table.csv`)
| Variant | SoH RMSE | SoH MAE | Monotonic Test Runs | Max Monotonicity Violation | Relative Impact vs SLAC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full SLAC** | **0.0573 ± 0.0138** | **0.0459 ± 0.0110** | **12 / 12 (100%)** | $+0.001350$ | Baseline ($0.0\%$) |
| **NoCrossAttn** | 0.0574 ± 0.0141 | 0.0462 ± 0.0111 | 12 / 12 (100%) | $+0.001352$ | $+0.0001$ ($+0.20\%$, negligible) |
| **NoHardGating** | 0.0596 ± 0.0197 | 0.0478 ± 0.0149 | 12 / 12 (100%) | $+0.001239$ | $+0.0023$ ($+4.01\%$) |
| **NoMonotonicity** | 0.1739 ± 0.1342 | 0.1463 ± 0.1117 | 0 / 12 (0%) | $-0.091053$ | **$+0.1166$ (+203.5% rounded / +203.8% unrounded)** |

### 3.3 Remaining Useful Life (RUL) Cell Breakdown (`artifacts/rul_validation_results.csv`)
| Cell ID | Latent-$D_t$ Extrap | Random Forest | XGBoost | StatDecomp | Raw SoH Extrap | Latent-$D_t$ vs Trees Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0005** (accelerating knee, 1.67×) | **4.94 ± 0.34** | 9.20 ± 0.07 | 9.21 ± 0.84 | 70.49 ± 0.00 | 84.60 ± 0.00 | **WIN** (~46% lower MAE than trees) |
| **B0006** (decelerating, 0.42×) | **16.99 ± 2.60** | 8.77 ± 0.05 | 8.27 ± 0.32 | 4.98 ± 0.00 | 7.73 ± 0.00 | **LOSS** (loses to tree models) |
| **B0018** (linear fade, 0.68×) | **7.35 ± 4.10** | 20.53 ± 0.29 | 20.23 ± 0.25 | 6.60 ± 0.00 | 5.80 ± 0.00 | **WIN** (~64% lower MAE than trees) |
| **B0007** (right-censored) | *NaN* | *NaN* | *NaN* | *NaN* | *NaN* | *Excluded (min cap 1.4005 Ah > 1.40 Ah)* |

*Head-to-head record vs tree ensembles across valid failure cells:* **2 Wins, 1 Loss**.

### 3.4 Intra-Cycle SoC Estimation (`artifacts/intracycle_soc_results.csv`)
* **Intra-Cycle GRU ([V, I, t]):** Overall RMSE = **0.0273 (2.73%)**, MAE = **0.0214**
  - B0005: $0.0272$ (2.72%)
  - B0006: $0.0513$ (5.13%)
  - B0007: $0.0145$ (1.45%)
  - B0018: $0.0163$ (1.63%)
* **Naive Coulomb Counting (Nominal 2.0 Ah):** Overall RMSE = **0.1450 (14.50%)**, MAE = **0.1127**
  - Terminal error reaches **41.37%** on B0006 as capacity fades.
  - Overall GRU error reduction: **-81.2%**.

### 3.5 Embedded Artifact Specs
* **Persistent Streaming BMS State:** Exactly **32 bytes** (`sizeof(pimdcnet_state_t)`: initial SoH [4B] + previous damage [4B] + 5-cycle damage history buffer [20B] + cycle count [4B]). Zero heap allocation.
* **Compiled ONNX Neural Graph:** `artifacts/pi_mdcnet_edge.onnx`: **318,034 bytes (~318 KB)**, opset 14, **67,013 float32 parameters**.
* **Sequential Parity Tolerance:** $< 1.19 \times 10^{-7}$ maximum absolute output difference over 50 steps.

---

## 4. Academic Defensibility Rules & Guardrails

1. **NO "First Ever" Claims:** Prior art established degradation/recovery decoupling (Qin et al. 2016) and battery monotonicity (monotonic PINNs). Our contribution is an architectural, end-to-end differentiable latent parameterization with hard gating.
2. **NO Claims of Formal Statistical Significance:** With $N=4$ cells ($N=3$ for RUL), formal statistical significance ($p < 0.05$) is mathematically impossible under the Wilcoxon signed-rank test ($p \ge 0.125$ for $N=4$; $p \ge 0.250$ for $N=3$). All inferential statistics are exploratory.
3. **Disambiguate Memory Claims:** 32 bytes refers strictly to persistent streaming state tracking in C, not on-device neural tensor inference. The neural graph is ~318 KB.
4. **Hardware Accuracy:** Target microcontroller is the ESP32-S3 (Xtensa LX7), never ARM Cortex-M.
5. **Report Honest RUL Win/Loss Record:** Latent-$D_t$ extrapolation excels on accelerating knees (B0005: 4.94 vs 9.20 trees), but loses on decelerating trajectories (B0006: 16.99 vs 8.27 trees). Always report both wins and losses.
6. **Report Single Labeled SoH RMSE Standard:** Use `0.0573 ± 0.0138 (pooled over 12 runs: 4 folds × 3 seeds)` as the primary metric, explicitly noting that across-fold cell variance is `0.0099`.
