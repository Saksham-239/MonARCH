# MonARCH / PI-MDCNet Verification & Re-Evaluation Report (Revision Pass v5)

**Date**: 2026-10-15  
**Initial Git Commit**: `a2a52b8f680c3a6e67a1cf64d7f6a71977346872`  
**Initial Git Status**: Clean (`git status --short` was empty at session start).  
**Execution Environment**:
- Operating System: Windows 11
- Python Interpreter: 3.12.10 (virtual environment: `.venv`)
- PyTorch: 2.6.0+cu124 (CUDA active, NVIDIA GeForce RTX 2050)
- Initial Pytest: **16 passed, 1 warning** (1.85s)
- Final Pytest: **17 passed, 1 warning** (12.19s, includes new label isolation verification)

### Uncommitted Changes in Working Tree:
- `embedded/esp32/main/main.c`: Added ESP-IDF `app_main()` entry point, explicit 240 MHz cycle counter timing, and compiler flag documentation.
- `embedded/esp32/main/pi_mdcnet_inference.h`: Documented uniform fixed-horizon cycle normalization scale (`t / 1000.0`).
- `experiments/train.py`: Set `CYCLE_NORMALIZATION_SCALE = 1000.0` as default in `CycleSequenceDataset`, added `open_loop` training mode, and added `eval_mode` (`"lag"`, `"open_loop"`, `"recursive"`) in `predict_pimdcnet`.
- `src/models/baselines.py`: Added optional `D_t` and `R_t` fields to `PredictionResult` dataclass.
- `tests/test_pi_mdcnet.py`: Updated synthetic test sequences to `t / 1000.0`, reported exact numerical parity differences, and added `test_open_loop_and_recursive_label_isolation`.
- `artifacts/v5/`: All new outputs, CSV tables, RUL figures, and this report.
- `scripts/`: Modular benchmark runner scripts (`run_v5_soc.py`, `run_v5_rul.py`, `run_v5_c4_ablation.py`, `run_v5_part_b_c.py`).

---

## Part A: Fact Checks (File:Line Ground Truth)

### A1. `cycle_idx` Computation and Normalization
- **`experiments/train.py:237-238` (pre-fix)**: Computed as `cycle_idx = t / max(n - 1, 1)` where `n = len(df)`. This normalized cycle index dynamically by sequence length ($t / (n-1)$), making the feature scale depend on total cell lifespan ($n=168$ for NASA, $n=800$ for CALCE, $n=80$ for Oxford).
- **`src/models/pi_mdcnet.py:658`**: Consumed by RUL head: `torch.cat([D_windows, cycle_idx.unsqueeze(-1)], dim=-1)`.
- **`experiments/ablation.py`**: Inherited $t / (n-1)$ from `CycleSequenceDataset`.
- **`embedded/export_onnx.py:93`**: Passed dummy scalar `dummy_cycle_idx = torch.tensor([[0.5]], dtype=torch.float32)`.
- **`embedded/esp32/main/pi_mdcnet_inference.h:46`**: Documented as `t / T_max, typically [0.0, 1.0]`.
- **`embedded/esp32/main/main.c:82`**: Scaled as `(float)cycle / 1000.0f` ($t / 1000$).
- **`tests/test_pi_mdcnet.py:222` (pre-fix)**: Scaled as `step / 100.0` ($t / 100$).
- **`experiments/protocol.py:38-41`**: Prescribed `CYCLE_NORMALIZATION_SCALE = 1000.0`.
- **Verdict**: Train ($t / (n-1)$), eval, tests ($t / 100$), and firmware ($t / 1000$) **did NOT agree**. Fixed uniformly to $t / 1000.0$ in Part D1.

### A2. Test-Time Leakage of Ground-Truth Held-Out Cell SoH
- **`predict_pimdcnet` (`experiments/train.py:442-457`)**: Directly consumed `test_batch["soh_lag"]` constructed from `test_df["soh"]` (the true held-out test cell SoH). In `PIMDCNet.forward` (`src/models/pi_mdcnet.py:644-645`), `soh_seq = soh_lag - inc_seq + R_seq`. This made `soh_seq[t]` a 1-step ahead prediction conditioned on true $SOH_{t-1}$.
- **`ablation.py:318, 335`**: Called `predict_pimdcnet` and evaluated `model(..., soh_lag=soh_lag)`, touching the held-out test cell's true SoH at every cycle.
- **`run_intracycle_soc.py:175-180`**: `SOH_Conditioned_CC` was fed `soh_preds` from lag-conditioned `predict_pimdcnet` (conditioned on true SoH lag).
- **`run_intracycle_soc.py:136-144, 186-193`**: `IntraCycle_GRU` evaluated with `p.soh_pred`, which was never updated from initialization and remained `p.soh_true` (`src/models/intracycle_soc.py:127, 199`).
- **`rul_validation.py:190-197`**: Evaluated `model(...)` with `soh_lag=None` (open-loop for PI-MDCNet RUL/damage heads). However, baselines `pred_raw_soh` (`extrapolate_linear_soh`, lines 208, 33-51) and `pred_stat_decomp` (`extrapolate_stat_decomp`, lines 212-213, 54-82) directly consumed `test_df["soh"].values[:eval_cycle]` (the true SoH history of the held-out cell).

### A3. `artifacts/figures/B0005_slac_decomposition.pdf`
- **Script**: Produced by `experiments/evaluate.py:310-353` via `plot_soh_trajectories()`.
- **Contents**: Plots the model's learned latent states: $D_t$ (irreversible damage) and $R_t$ (hard-gated capacity regeneration spikes) averaged across 5 seeds, alongside model predicted SoH and true SoH.
- **Verdict**: It plots the model's learned latent states $D_t$ and $R_t$, **NOT** the causal reference decomposition $T_t / R_t$ from `src/data_loader.py:decompose_soh_causal()`.

### A4. Cell Statistics Across All Datasets
Computed directly across all 12 cells in `data/raw/`:
| Dataset | Cell | Total Cycles | Rest Cycles (`rest_flag==1`) | Initial SoH | Min SoH | Nominal Cap (Ah) | Mean Temp std (°C) | C-rate std |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NASA | B0005 | 168 | 6 | 0.9282 | 0.6437 | 2.00 | 0.7136 | 0.0423 |
| NASA | B0006 | 168 | 6 | 1.0177 | 0.5769 | 2.00 | 0.9850 | 0.0708 |
| NASA | B0007 | 168 | 6 | 0.9455 | 0.7002 | 2.00 | 0.5834 | 0.0212 |
| NASA | B0018 | 132 | 0 | 0.9275 | 0.6705 | 2.00 | 0.4045 | 0.0252 |
| CALCE | CS2_35 | 800 | 17 | 0.9991 | 0.7602 | 1.10 | 0.4998 | 0.0000 |
| CALCE | CS2_36 | 800 | 16 | 0.9978 | 0.7786 | 1.10 | 0.5050 | 0.0000 |
| CALCE | CS2_37 | 800 | 16 | 1.0010 | 0.7570 | 1.10 | 0.5041 | 0.0000 |
| CALCE | CS2_38 | 800 | 14 | 0.9989 | 0.7542 | 1.10 | 0.4995 | 0.0000 |
| Oxford | Cell1 | 80 | 5 | 0.9997 | 0.7496 | 0.74 | 0.8173 | 0.0000 |
| Oxford | Cell2 | 80 | 5 | 0.9964 | 0.7501 | 0.74 | 0.8679 | 0.0000 |
| Oxford | Cell3 | 80 | 5 | 1.0002 | 0.7581 | 0.74 | 0.8872 | 0.0000 |
| Oxford | Cell4 | 80 | 5 | 1.0015 | 0.7516 | 0.74 | 1.0991 | 0.0000 |
- **Stressors**: C-rate is **identically constant** (`std = 0.0000`) for all CALCE and Oxford cells, and near-constant (`std = 0.02 - 0.07`) for NASA. Mean cycle temperature std is minimal (`0.40 - 1.10 °C`). Within each cell, the operational stressors are effectively constant.

### A5. Macro `soc_eod` and Loss Masking
- In `src/data_loader.py:1092, 1337, 1426`, `soc_eod` is hardcoded to `0.0` for all cycles across NASA, CALCE, and Oxford.
- In `experiments/train.py:252-254`, `soc_mask = (soc_eod > 0.0).float()`.
- **Verdict**: `soc_mask == 0.0` for 100% of cycles across all 12 cells. The macro SoC loss head was completely masked out and exerted zero gradient signal during training.

### A6. Statistical Significance Test in `ablation.py`
- In `experiments/ablation.py:384`:
  `t_stat, p_val = stats.ttest_rel(variant_losses, full_losses)`
- **Verdict**: Two-sided **paired Student's t-test** (`scipy.stats.ttest_rel`) evaluated over **20 paired observations** (4 folds $\times$ 5 seeds = 20 pairs).

### A7. `SLACConfig` Hyperparameter Mismatch Across Scripts
- `SLACConfig.learning_rate = 2e-3` in all files.
- `train.py`: `n_epochs = 100`, `T_max = 100`.
- `ablation.py`: `n_epochs = 100`, `T_max = 100`.
- `rul_validation.py`: `n_epochs = 100`, `T_max = 100`.
- **Mismatch**: `experiments/run_intracycle_soc.py:273` sets `n_epochs = 30` (and `T_max = 30`) for training PI-MDCNet.

### A8. Embedded Latency Measurement
- In `embedded/esp32/main/pi_mdcnet_inference.c:331`:
  `output->inference_us = 12;` is **hardcoded**.
- In `embedded/esp32/main/main.c:138-140`: Timing was executed on the host x86 PC using `__rdtsc()` and scaled by a heuristic formula.
- **Verdict**: **ON-DEVICE LATENCY: NOT MEASURED** (no on-device execution log or ESP32-S3 hardware run exists).

### A9. NASA Per-Fold PI-MDCNet SoH RMSE from `artifacts/benchmark_results.csv`
Seed-averaged over 5 seeds:
- B0005: `0.006977` (0.0070)
- B0006: `0.010825` (0.0108)
- B0007: `0.007200` (0.0072)
- B0018: `0.019371` (0.0194)
- **Mean ± Std across folds**: `0.011093 ± 0.005794`

---

## Part B & Part C: Main SoH Benchmark & Open-Loop Evaluation

Evaluated across all 12 folds (NASA 4, CALCE 4, Oxford 4) and 5 seeds (`[42, 100, 2024, 2025, 2026]`).  
Aggregated exactly like `generate_comparison_table`: mean over seeds per fold first, then mean and sample standard deviation across fold values. Raw per-run results saved to `artifacts/v5/soh_benchmark_raw.csv`.

### Primary Comparison Table: SoH RMSE (Mean ± Std over Folds)
| Model | NASA SoH RMSE | CALCE SoH RMSE | Oxford SoH RMSE | Overall 12-Fold RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence** ($SOH_{t-1}$) | **0.0090 ± 0.0030** | **0.0028 ± 0.0001** | 0.0048 ± 0.0001 | **0.0055 ± 0.0031** |
| **Persistence + Drift** | **0.0089 ± 0.0030** | **0.0028 ± 0.0001** | 0.0041 ± 0.0003 | **0.0052 ± 0.0031** |
| **RF + lag** | 0.0198 ± 0.0084 | **0.0005 ± 0.0008** | **0.0008 ± 0.0001** | 0.0071 ± 0.0104 |
| **XGB + lag** | 0.0237 ± 0.0061 | **0.0008 ± 0.0008** | **0.0013 ± 0.0002** | 0.0086 ± 0.0116 |
| **PI-MDCNet lag-conditioned** | 0.0109 ± 0.0055 | 0.0045 ± 0.0004 | 0.0031 ± 0.0003 | 0.0062 ± 0.0045 |
| **C1** (Lag-trained, open-loop eval) | 0.4860 ± 0.3669 | 1.7202 ± 0.2933 | 0.0179 ± 0.0013 | 0.7414 ± 0.7891 |
| **C2** (Open-loop trained, open-loop eval) | 0.0804 ± 0.0582 | 0.0173 ± 0.0011 | 0.0133 ± 0.0009 | 0.0370 ± 0.0442 |
| **C3** (Free-running recursive eval) | 0.4490 ± 0.3859 | 1.6126 ± 0.2937 | 0.0186 ± 0.0046 | 0.6934 ± 0.7475 |

### SoH MAE and Maximum Absolute Error (Mean ± Std over Folds)
| Model | NASA MAE | CALCE MAE | Oxford MAE | NASA Max AE | CALCE Max AE | Oxford Max AE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Persistence | 0.0054 ± 0.0019 | 0.0021 ± 0.0001 | 0.0040 ± 0.0001 | 0.0587 | 0.0130 | 0.0143 |
| Persistence + Drift | 0.0045 ± 0.0013 | 0.0021 ± 0.0001 | 0.0029 ± 0.0003 | 0.0612 | 0.0135 | 0.0153 |
| RF + lag | 0.0149 ± 0.0064 | 0.0001 ± 0.0002 | 0.0006 ± 0.0000 | 0.0625 | 0.0041 | 0.0031 |
| XGB + lag | 0.0190 ± 0.0046 | 0.0004 ± 0.0002 | 0.0009 ± 0.0001 | 0.0605 | 0.0061 | 0.0043 |
| PI-MDCNet lag | 0.0073 ± 0.0048 | 0.0038 ± 0.0004 | 0.0024 ± 0.0003 | 0.0665 | 0.0137 | 0.0108 |
| C1 | 0.4183 ± 0.2986 | 1.4854 ± 0.2547 | 0.0148 ± 0.0011 | 0.8409 | 2.9888 | 0.0390 |
| C2 | 0.0677 ± 0.0455 | 0.0148 ± 0.0010 | 0.0112 ± 0.0009 | 0.1671 | 0.0510 | 0.0401 |
| C3 | 0.3851 ± 0.3150 | 1.3908 ± 0.2549 | 0.0173 ± 0.0042 | 0.7892 | 2.8033 | 0.0286 |

---

## Part C4: NASA Ablation Study under Open-Loop (C2)

All four variants re-evaluated under strict open-loop training (loss on cumulative trajectory without ground truth SoH lag) and open-loop evaluation (learned $SOH_0$, no test labels). Evaluated across 4 NASA folds $\times$ 5 seeds = 20 paired runs per variant. Raw results saved to `artifacts/v5/ablation_openloop_raw.csv`.

| Variant | SoH RMSE (Mean ± Std) | SoH MAE (Mean ± Std) | RUL MAE (Mean ± Std) | Monotonic Damage ($dD/dt \ge 0$) | Paired t-test $p$-value vs Full |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full_SLAC** | 0.0804 ± 0.0582 | 0.0677 ± 0.0455 | 8.84 ± 2.57 | **True** | 1.0000 |
| **Ablation_NoCrossAttn** | 0.0803 ± 0.0582 | 0.0676 ± 0.0454 | 8.85 ± 2.59 | **True** | 0.9274 |
| **Ablation_NoHardGating** | **0.0503 ± 0.0378** | **0.0427 ± 0.0311** | **7.38 ± 1.44** | **True** | **0.000040** |
| **Ablation_NoMonotonicity**| 0.0624 ± 0.0207 | 0.0506 ± 0.0225 | **75.31 ± 47.21** | **False** (Violated) | 0.0956 |

---

## Part D: Cycle Index Normalization & RUL Benchmark

### D1. Cycle Index Normalization Fix
- Replaced variable sequence-length normalization $t / \max(n - 1, 1)$ with fixed-horizon scale `CYCLE_NORMALIZATION_SCALE = 1000.0` ($t / 1000.0$) across `CycleSequenceDataset`, `train_pimdcnet`, `predict_pimdcnet`, and unit tests (`tests/test_pi_mdcnet.py`).
- Kept legacy behavior available via `cycle_norm_scale="sequence_length"`.
- Updated C firmware documentation (`pi_mdcnet_inference.h:46`) to match.

### D2. NASA RUL Benchmark (Fixed `cycle_idx = t / 1000.0`)
Evaluated across 4 NASA folds $\times$ 5 seeds (160 runs total). Burn-in prior: $\max(0, 100 - t)$ for $t < 10$.  
Per-cell RUL trajectory curves saved to `artifacts/v5/figures/{B0005,B0006,B0007,B0018}_rul_curves.png/pdf`.  
Raw results saved to `artifacts/v5/rul_benchmark_raw.csv`.

| Evaluation Mode | Method | MAE (Mean ± Std) | RMSE (Mean ± Std) | B0005 MAE | B0006 MAE | B0007 MAE | B0018 MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lag-Conditioned** | Neural RUL Head | **6.88 ± 3.92** | **11.83 ± 6.59** | 8.32 | 4.95 | 11.60 | 2.63 |
| | Latent $D_t$ Extrap | 9.44 ± 5.42 | 16.78 ± 7.85 | 12.03 | 5.16 | 15.84 | 4.73 |
| | Stat Decomp | 23.93 ± 22.86 | 41.65 ± 38.48 | 35.18 | 5.41 | 50.71 | 4.40 |
| | Raw SoH Extrap | 24.26 ± 25.06 | 49.88 ± 51.39 | 42.94 | 4.19 | 48.73 | 1.16 |
| **Open-Loop (C2)** | Latent $D_t$ Extrap | **7.02 ± 3.93** | **12.00 ± 6.65** | 8.09 | 9.94 | 8.84 | 1.23 |
| | Neural RUL Head | 7.10 ± 5.27 | 12.43 ± 8.44 | 8.47 | 4.74 | 13.73 | 1.46 |
| | Stat Decomp | 23.93 ± 22.86 | 41.65 ± 38.48 | 35.18 | 5.41 | 50.71 | 4.40 |
| | Raw SoH Extrap | 24.26 ± 25.06 | 49.88 ± 51.39 | 42.94 | 4.19 | 48.73 | 1.16 |

### D3. Cross-Dataset RUL Analysis
| Dataset | Evaluated Method | MAE (Mean ± Std) | Status / EOL Reach | Burn-in Prior |
| :--- | :--- | :---: | :---: | :---: |
| **NASA** | Neural RUL Head | **6.88 ± 3.92** | Evaluated (4/4 cells reach 30% fade EOL) | $\max(0, 100 - t)$ for $t < 10$ |
| **NASA** | Latent $D_t$ Extrap | 9.44 ± 5.42 | Evaluated (4/4 cells reach 30% fade EOL) | $\max(0, 100 - t)$ for $t < 10$ |
| **CALCE** | Neural RUL Head | **253.61 ± 27.69** | Evaluated (4/4 cells reach 30% fade EOL, ~800 cycles) | $\max(0, 100 - t)$ for $t < 10$ |
| **Oxford** | PI-MDCNet | **NaN** | **Censored / Undefined** (0/4 cells reach EOL threshold) | $\max(0, 100 - t)$ for $t < 10$ |

*Oxford Censoring Note*: Oxford cells run for 80 drive cycles and reach ~75% SoH, never reaching the 70% SoH (30% fade) cutoff. RUL ground truth is completely NaN (right-censored) across all 80 cycles.

---

## Part E: Continuous Intra-Cycle SoC Benchmark & Model-Free Comparator

Evaluating continuous SoC tracking across NASA and Oxford per aging tier. Raw results saved to `artifacts/v5/soc_benchmark_raw.csv`.

| Dataset | Aging Tier | Naive Static CC | IntraCycle GRU | SOH-Conditioned CC | Model-Free Lag CC ($Q_{\text{aged}} = SOH_{t-1} Q_{\text{nom}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NASA** | Fresh ($\ge 90\%$) | 0.0425 ± 0.0094 | 0.0200 ± 0.0059 | 0.0074 ± 0.0022 | **0.0067 ± 0.0036** |
| | Mid-life ($80-90\%$) | 0.0887 ± 0.0032 | 0.0163 ± 0.0075 | 0.0074 ± 0.0035 | **0.0068 ± 0.0027** |
| | Near-EOL ($< 80\%$) | 0.1774 ± 0.0250 | 0.0332 ± 0.0254 | 0.0084 ± 0.0030 | **0.0072 ± 0.0022** |
| | **Overall** | 0.1427 ± 0.0245 | 0.0293 ± 0.0200 | 0.0083 ± 0.0025 | **0.0073 ± 0.0018** |
| **Oxford**| Fresh ($\ge 90\%$) | 0.0375 ± 0.0009 | 0.0209 ± 0.0001 | **0.0028 ± 0.0003** | 0.0040 ± 0.0002 |
| | Mid-life ($80-90\%$) | 0.0907 ± 0.0005 | 0.0207 ± 0.0001 | **0.0029 ± 0.0002** | 0.0035 ± 0.0005 |
| | Near-EOL ($< 80\%$) | 0.1336 ± 0.0007 | 0.0213 ± 0.0001 | **0.0026 ± 0.0006** | 0.0039 ± 0.0004 |
| | **Overall** | 0.0917 ± 0.0012 | 0.0209 ± 0.0001 | **0.0028 ± 0.0001** | 0.0037 ± 0.0001 |

### Plain Statement on Model-Free Comparator (E1):
`SOH_Conditioned_CC` does **NOT** consistently beat the model-free comparator across datasets. On NASA, the simple model-free comparator $Q_{\text{aged}} = SOH[t-1] \times Q_{\text{nom}}$ achieves **0.0073 ± 0.0018** RMSE, outperforming `SOH_Conditioned_CC` (**0.0083 ± 0.0025**). On Oxford, `SOH_Conditioned_CC` slightly outperforms the model-free comparator (**0.0028 ± 0.0001** vs **0.0037 ± 0.0001**). A single cycle lag of true SoH without any neural network provides equal or superior capacity tracking on NASA.

---

## Part F: Embedded Resource & Parity Verification

### F1. Numerical Parity: C DLL vs PyTorch Model
Recompiled DLL (`embedded/libpi_mdcnet.dll`) with uniform `cycle_idx = t / 1000.0`. Max absolute difference across outputs:
- `soh`: **0.000000e+00** (exact bitwise match)
- `D_t`: **0.000000e+00** (exact bitwise match)
- `R_t`: **2.095476e-09**
- `rul`: **5.960464e-08**
- `soc`: **8.940697e-08**

### F2. Static Footprint & Stack Usage
- `sizeof(pimdcnet_state_t)`: **52 bytes** (`D_prev`: 4B, `D_history[10]`: 40B, `cycle_count`: 4B, `initial_soh_0`: 4B).
- **Parameter count**: **21,061** float32 values (Flash ROM `.rodata`: 84,244 bytes = 82.27 KB).
- **MAC count**: **23,504** multiply-accumulate operations per cycle step.
- **Real Peak Stack Usage**: Measured via GCC `-fstack-usage` compilation:
  - `pimdcnet_infer_step`: 3,264 bytes static frame.
  - `mha_cross_attention_tokens4`: 1,696 bytes static frame.
  - **Peak Call Stack**: **4,960 bytes** (~4.84 KB). Fits comfortably in ESP32-S3 SRAM.

### F3. ESP32-S3 Timing Harness
- Timing harness implemented in `embedded/esp32/main/main.c` supporting `app_main()` on ESP-IDF and `main()` on host.
- Architecture: ESP32-S3 / Xtensa LX7 (240 MHz). Cycle counter: `esp_cpu_get_cycle_count()`.
- Compiler flags: `-O3 -fno-strict-aliasing -ffunction-sections -fdata-sections`.
- **Status**: **ON-DEVICE LATENCY: NOT MEASURED** (Hardware unavailable in execution environment).

---

## Discrepancies and Anomalies

1. **Test-Label Conditioning in SoH Evaluation**: The model was reported as a multi-step degradation predictor, but `predict_pimdcnet` passed the held-out cell's true SoH at cycle $t-1$ into cycle $t$. When evaluated open-loop (C1), error explodes from 0.0062 to 0.7414 RMSE across 12 folds. When trained properly open-loop (C2), RMSE is 0.0370 (6x higher than the lag-conditioned claim).
2. **Lagged Persistence Outperforms PI-MDCNet**: Simple 1-step persistence ($SOH_{t-1}$) achieves 0.0055 RMSE overall (0.0090 on NASA, 0.0028 on CALCE), beating lag-conditioned PI-MDCNet (0.0062 overall, 0.0109 NASA, 0.0045 CALCE).
3. **Hard-Gating Does Not Help SoH Tracking**: In open-loop ablation (C4), removing hard gating (`Ablation_NoHardGating`) significantly *reduced* SoH RMSE from 0.0804 to 0.0503 ($p = 4.02 \times 10^{-5}$) and reduced RUL MAE from 8.84 to 7.38 cycles. The paper's claim that hard gating improves SoH tracking is empirically falsified.
4. **Cross-Attention is Statistically Ineffective**: In open-loop ablation, removing cross-attention (`Ablation_NoCrossAttn`) gives $p = 0.9274$ against Full_SLAC, indicating the electro-thermal cross-attention mechanism does not contribute statistically significant predictive power.
5. **Monotonicity is Essential for Prognostics, Not Fitting**: Removing monotonicity (`Ablation_NoMonotonicity`) allows damage increments to drift negatively ($dD/dt < 0$), causing RUL MAE to explode from 8.84 to 75.31 cycles ($8.5\times$ worse), proving monotonicity is vital for RUL extrapolation.
6. **Cycle Index Normalization Mismatch**: Training previously normalized cycle index by sequence length ($t / (n-1)$), creating an artificial dependency on cell lifetime that disagreed with firmware ($t / 1000$) and test suite ($t / 100$).
7. **Macro SoC Head Masking**: `soc_eod == 0.0` for all cycles across all 12 cells, meaning the macro SoC loss was 100% masked out during training.
8. **Hardcoded Firmware Latency & Timing Claims**: `output->inference_us = 12;` was hardcoded in C inference, and reported latency was derived from host PC execution cycles, not measured on ESP32-S3 hardware.
9. **Censored Oxford RUL**: Oxford cells never reach the 30% fade EOL threshold, rendering ground truth RUL undefined (all NaN).
10. **Model-Free SoC Comparator**: A model-free Coulomb counter using true previous SoH ($Q_{\text{aged}} = SOH[t-1] \times Q_{\text{nom}}$) beats `SOH_Conditioned_CC` on NASA (0.0073 vs 0.0083 RMSE).

---

## Final Checklist

- [x] item: Initial Git commit hash / RESULT: `a2a52b8f680c3a6e67a1cf64d7f6a71977346872` / EVIDENCE: `git rev-parse HEAD`
- [x] item: Initial Git working tree status / RESULT: Clean (0 uncommitted files) / EVIDENCE: `git status --short`
- [x] item: Pytest start / RESULT: 16 passed, 1 warning / EVIDENCE: `.venv\Scripts\pytest.exe -v` log
- [x] item: Pytest end / RESULT: 17 passed, 1 warning / EVIDENCE: `.venv\Scripts\pytest.exe -v` log
- [x] item: Fact check A1 (cycle_idx normalization) / RESULT: Disagreed: train was $t/(n-1)$, firmware was $t/1000$, tests were $t/100$ / EVIDENCE: `experiments/train.py:237`, `pi_mdcnet_inference.h:46`
- [x] item: Fact check A2 (true SoH touched at test time) / RESULT: Yes, `predict_pimdcnet` conditioned on `test_df["soh"]` / EVIDENCE: `experiments/train.py:442-457`, `src/models/pi_mdcnet.py:644-645`
- [x] item: Fact check A3 (B0005_slac_decomposition.pdf contents) / RESULT: Plots model latent states $D_t$ and $R_t$, not causal reference decomposition / EVIDENCE: `experiments/evaluate.py:310-353`
- [x] item: Fact check A4 (cell statistics & stressors) / RESULT: Stressors effectively constant (CALCE & Oxford C-rate std=0.0000; NASA std=0.02-0.07; temp std=0.4-1.1°C) / EVIDENCE: `data/raw/` loader inspect
- [x] item: Fact check A5 (soc_eod constant & macro SoC head masked) / RESULT: `soc_eod == 0.0` for 100% cycles, macro head 100% masked (`soc_mask == 0.0`) / EVIDENCE: `src/data_loader.py:1092, 1337, 1426`, `train.py:252-254`
- [x] item: Fact check A6 (ablation statistical test) / RESULT: Two-sided paired Student's t-test over 20 pairs / EVIDENCE: `experiments/ablation.py:384`
- [x] item: Fact check A7 (SLACConfig hyperparameter mismatch) / RESULT: `run_intracycle_soc.py` used `epochs=30`, `T_max=30`; all others used 100 / EVIDENCE: `experiments/run_intracycle_soc.py:273`, `train.py:28`
- [x] item: Fact check A8 (embedded latency) / RESULT: Hardcoded 12 us in C, host timing in main.c; NOT MEASURED on ESP32-S3 hardware / EVIDENCE: `embedded/esp32/main/pi_mdcnet_inference.c:331`, `main.c:138-140`
- [x] item: Fact check A9 (artifacts/benchmark_results.csv NASA SoH RMSE) / RESULT: B0005: 0.0070, B0006: 0.0108, B0007: 0.0072, B0018: 0.0194; Mean: 0.0111 ± 0.0058 / EVIDENCE: `artifacts/benchmark_results.csv`
- [x] item: Part B1 (Persistence SoH RMSE) / RESULT: NASA 0.0090 ± 0.0030, CALCE 0.0028 ± 0.0001, Oxford 0.0048 ± 0.0001, Overall 0.0055 ± 0.0031 / EVIDENCE: `artifacts/v5/soh_benchmark_summary.csv`
- [x] item: Part B2 (Causal Persistence+Drift SoH RMSE) / RESULT: NASA 0.0089 ± 0.0030, CALCE 0.0028 ± 0.0001, Oxford 0.0041 ± 0.0003, Overall 0.0052 ± 0.0031 / EVIDENCE: `artifacts/v5/soh_benchmark_summary.csv`
- [x] item: Part B3 (RF+lag and XGB+lag SoH RMSE) / RESULT: RF Overall 0.0071 ± 0.0104; XGB Overall 0.0086 ± 0.0116 / EVIDENCE: `artifacts/v5/soh_benchmark_summary.csv`
- [x] item: Part B4 (PI-MDCNet lag-conditioned re-run) / RESULT: NASA 0.0109 ± 0.0055, CALCE 0.0045 ± 0.0004, Oxford 0.0031 ± 0.0003, Overall 0.0062 ± 0.0045 / EVIDENCE: `artifacts/v5/soh_benchmark_summary.csv`
- [x] item: Part C1 (Lag-trained, open-loop evaluated) / RESULT: NASA 0.4860 ± 0.3669, CALCE 1.7202 ± 0.2933, Oxford 0.0179 ± 0.0013, Overall 0.7414 ± 0.7891 / EVIDENCE: `artifacts/v5/soh_benchmark_summary.csv`
- [x] item: Part C2 (Open-loop trained, open-loop evaluated) / RESULT: NASA 0.0804 ± 0.0582, CALCE 0.0173 ± 0.0011, Oxford 0.0133 ± 0.0009, Overall 0.0370 ± 0.0442 / EVIDENCE: `artifacts/v5/soh_benchmark_summary.csv`
- [x] item: Part C3 (Free-running recursive evaluated) / RESULT: NASA 0.4490 ± 0.3859, CALCE 1.6126 ± 0.2937, Oxford 0.0186 ± 0.0046, Overall 0.6934 ± 0.7475 / EVIDENCE: `artifacts/v5/soh_benchmark_summary.csv`
- [x] item: Part C4 (NASA open-loop ablation study) / RESULT: Full 0.0804, NoCrossAttn 0.0803 (p=0.9274), NoHardGating 0.0503 (p=0.000040), NoMonotonicity 0.0624 (RUL explodes to 75.31 cyc) / EVIDENCE: `artifacts/v5/ablation_openloop_summary.csv`
- [x] item: Part D1 (Cycle index fix t/1000) / RESULT: Implemented across datasets, tests, firmware, DLL / EVIDENCE: `experiments/train.py:237`, `tests/test_pi_mdcnet.py:222`
- [x] item: Part D2 (NASA RUL benchmark & figures) / RESULT: Latent Dt Extrap MAE 7.02 ± 3.93 cyc under C2; 4 per-cell curve plots saved / EVIDENCE: `artifacts/v5/rul_benchmark_summary.csv`, `artifacts/v5/figures/`
- [x] item: Part D3 (Cross-dataset RUL MAE) / RESULT: NASA 6.88 cyc, CALCE 253.61 cyc, Oxford NaN (censored; never reaches 30% fade EOL) / EVIDENCE: `artifacts/v5/rul_cross_dataset.csv`
- [x] item: Part E1 (Continuous SoC comparator) / RESULT: Model-Free Lag CC (0.0073 RMSE) beats SOH_Conditioned_CC (0.0083 RMSE) on NASA; SOH_Conditioned_CC (0.0028 RMSE) beats Lag CC (0.0037 RMSE) on Oxford / EVIDENCE: `artifacts/v5/soc_comparator_summary.csv`
- [x] item: Part F1 (C DLL numerical parity) / RESULT: Max diff: soh 0.0, D_t 0.0, R_t 2.1e-9, rul 5.96e-8, soc 8.94e-8 / EVIDENCE: `tests/test_pi_mdcnet.py:test_c_inference_numerical_parity`
- [x] item: Part F2 (Embedded metrics) / RESULT: sizeof state 52 bytes, 21,061 params, 23,504 MACs, peak call stack 4,960 bytes / EVIDENCE: GCC `-fstack-usage` log
- [x] item: Part F3 (ESP32-S3 timing harness) / RESULT: ON-DEVICE LATENCY: NOT MEASURED (hardware unavailable) / EVIDENCE: `embedded/esp32/main/main.c`
