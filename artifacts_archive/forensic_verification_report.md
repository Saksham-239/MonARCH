# PI-MDCNet: Comprehensive Forensic Audit & Publication Defensibility Report

**Project:** Physics-Informed Multi-Domain Convolutional Network (PI-MDCNet v4 / SLAC)  
**Date:** September 2026  
**Auditor:** Antigravity Pair-Programming Agent (Advanced Agentic Coding)  
**Objective:** Establish complete technical truthfulness, empirical traceability, and academic defensibility before an adversarial review panel or skeptical faculty committee.

---

## 1. Executive Summary of Forensic Findings

This forensic audit reviewed the entire PI-MDCNet codebase, raw NASA PCoE data files (`B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`), execution scripts, C99 embedded source files, ONNX runtime export pipelines, and benchmark logs.

### Key Corrections Applied
1. **NASA PCoE Protocol Cutoff Voltages:** Corrected the erroneous claim that B0007 had a 2.7 V discharge cutoff. Verification from raw voltage arrays and official NASA PCoE specifications confirms:
   - **B0005:** 2.7 V cutoff
   - **B0006:** 2.5 V cutoff
   - **B0007:** 2.2 V cutoff (*corrected from 2.7 V*)
   - **B0018:** 2.5 V cutoff
2. **Embedded Target Architecture:** Corrected "ARM Cortex-M" references for the ESP32-S3 target in `main.c`, `pi_mdcnet_inference.h`, and `export_onnx.py`. The ESP32-S3 utilizes a dual-core 32-bit **Xtensa LX7** processor.
3. **Memory Footprint Disambiguation:** Disambiguated the "32-byte RAM" claim. Exactly 32 bytes corresponds solely to the persistent streaming BMS state structure (`pimdcnet_state_t`: `soh_0` [4B] + `D_prev` [4B] + `D_history[5]` [20B] + `cycle_count` [4B]). The full neural model graph (`pi_mdcnet_edge.onnx`) is ~318 KB (~78k parameters). The C demonstration is an analytical streaming state-update engine with zero heap allocation, distinct from full on-device tensor execution.
4. **SoC Reference Truth Terminology:** Corrected claims describing retrospective Coulomb integration as "independently measured SoC ground truth." The target is a **retrospective cycle-normalized SoC reference** ($1 - q(t)/Q_{\text{cycle}}$), where $Q_{\text{cycle}}$ is strictly withheld from the model during inference ($[V(t), I(t), t]$ inputs only).
5. **Stressor Conditioning Input Vector:** Confirmed that active model inputs for $s_t$ are strictly 2-dimensional: $[ \text{mean\_temp\_cycle}, \text{c\_rate} ]$. Claims that $s_t$ contains Depth-of-Discharge (DoD) were factually false and have been eliminated.
6. **Cross-Attention Novelty Downgrade:** Ablation evidence reveals an isolated gain of only $0.0001$ RMSE ($0.0573 \pm 0.0138$ vs. $0.0574 \pm 0.0141$). Cross-attention is treated as an architectural fusion mechanism, not a primary performance driver.
7. **Small-Sample Statistical Inference:** With $N=4$ battery trajectories (and $N=3$ valid EOL cells for RUL), formal statistical significance ($p < 0.05$) is mathematically impossible under the two-sided Wilcoxon signed-rank test (minimum possible $p = 0.125$ for $N=4$ and $p = 0.250$ for $N=3$). Repeated initializations (seeds) are not independent battery samples. All inferential statistics are reported as exploratory.

---

## 2. Forensic Verification Table (All 22 Audited Claims)

| Claim ID | Original Claim / Domain | Status | Evidence in Code & Data | Required Correction | Remaining Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C-01** | First model to separate irreversible degradation and reversible recovery | **FALSE** | Qin et al. (2016) explicitly decoupled degradation and rest-induced recovery using particle filtering. | Frame contribution as end-to-end differentiable latent decomposition ($SOH_t = SOH_0 - D_t + R_t$). | Prior literature established the physical concept. |
| **C-02** | First battery model using monotonicity | **FALSE** | Extensive literature (PINNs, health indicators, penalty losses) enforces battery monotonicity. | Frame as architectural/parametric enforcement ($\Delta D_t > 0$) vs. soft penalty loss. | Parametric monotonicity has precedent in neural networks. |
| **C-03** | First model using rest-time regeneration | **FALSE** | Qin et al. (2016, 2019) model recovery amplitude as a function of preceding rest time. | Acknowledge prior rest-time models; highlight hard-gating mechanism ($R_t = 0$ during cycling). | Rest duration remains an empirical proxy. |
| **C-04** | Stressor-conditioned degradation is unprecedented | **FALSE** | Arrhenius and throughput aging models routinely condition on temperature and C-rate. | Frame as routing operating stressors ($s_t$) directly to positive damage-rate parameterization. | Evaluated only across small ambient variations (24°C chamber). |
| **C-05** | Cross-attention is a major accuracy driver | **FALSE** | `ablation_table.csv`: Full SLAC = $0.0573 \pm 0.0138$, NoCrossAttn = $0.0574 \pm 0.0141$ ($\Delta = 0.0001$). | Treat cross-attention as an architectural mechanism; explicitly note isolated gain is negligible. | No statistical significance over feature concatenation. |
| **C-06** | $\Delta D_t = \lambda \text{Softplus}(f_\theta(s_t)) > 0$ guarantees monotonic cumulative damage | **VERIFIED** | `pi_mdcnet.py`: $\lambda = 0.003 > 0$, $\text{Softplus}(u) > 0$, $D_t = \text{cumsum}(\Delta D_t) > D_{t-1}$. | Maintain mathematical definition and code verification; 12/12 monotonic in ablation. | Monotonicity of $D_t$ does not force total $SOH_t$ to be monotonic (due to $R_t$). |
| **C-07** | $SOH_t = SOH_0 - D_t + R_t$ decomposition | **VERIFIED** | `pi_mdcnet.py` L624: `soh_seq = self.soh_0 - D_seq + R_seq`. Reconstruction error $< 10^{-6}$. | Emphasize integrated recurrent latent-state formulation rather than inventing the decomposition. | Identifiability relies on decorrelation between $s_t$ and rest features. |
| **C-08** | Stressor vector $s_t = [\text{C-rate}, \text{Temp}, \text{DoD}]$ | **PARTIAL** | `train.py` L65: `STRESSOR_COLS = ["mean_temp_cycle", "c_rate"]`. Dimension is strictly 2. No DoD. | Correct documentation to state $s_t$ is 2D. Note DoD/throughput was excluded to avoid target leakage. | Only 2 macro stressor features used. |
| **C-09** | Hard gating of regeneration ($R_t = 0$ during cycling) | **VERIFIED** | `pi_mdcnet.py` L425: $R_t = \text{rest\_flag}_t \cdot R_{\text{raw}}$. Non-rest cycles have $R_t \equiv 0.0$. | Maintain property; clarify that hard gating is an explicit design choice, not an unprecedented invention. | Binary gating does not capture multi-cycle relaxation decay dynamics. |
| **C-10** | Intra-cycle SoC GRU RMSE is 2.73% | **VERIFIED** | `eval_naive_soc.py` & `intracycle_soc_results.csv`: B0005=2.72%, B0006=5.13%, B0007=1.45%, B0018=1.63%. | Preserve exact numbers; explain per-cell dispersion. | Tested under constant 2A CC discharge profiles. |
| **C-11** | SoC target is independently measured ground truth | **FALSE** | `intracycle_soc.py`: $SoC_{\text{ref}}(t) = 1 - q(t) / Q_{\text{cycle}}$, Coulomb integration over same current. | Rename to "retrospective cycle-normalized SoC reference". Confirm $Q_{\text{cycle}}$ withheld at inference. | Not an independent electrochemical probe measurement. |
| **C-12** | Voltage profile alone causes SoC improvement | **UNSUPPORTED** | Inputs are $[V(t), I(t), t]$. Without feature ablation, isolating voltage alone is unsupported. | State: "GRU has access to voltage and temporal information unavailable to open-loop Coulomb counting." | Feature ablation shows joint $[V, I, t]$ synergy. |
| **C-13** | Coulomb counting maximum terminal error reaches 41.37% | **VERIFIED** | `eval_naive_soc.py`: B0006 maximum error = $0.4137$ at end of life. | Rephrase: "Fixed nominal-capacity normalization causes terminal bias as capacity fades below 2.0 Ah." | Expected failure mode of open-loop uncalibrated Coulomb counting. |
| **C-14** | NASA cutoff voltages: B0005=2.7V, B0006=2.5V, B0007=2.7V, B0018=2.5V | **PARTIAL** | Raw data in `B0007.mat` shows min voltage = 1.74V, median = 2.10V. Official spec cutoff is **2.2 V**. | Correct B0007 to **2.2 V** in all documentation, code comments, and presentation materials. | Historical typo in documentation corrected. |
| **C-15** | Strict LOBO-CV protocol without data leakage | **VERIFIED** | `train.py`: `FeatureScaler` fit strictly on training cells; test cell strictly held out across all 4 folds. | Document protocol explicitly; verify zero leakage in scaling, windowing, and label extraction. | Cross-validation limited to 4 cells. |
| **C-16** | Causal RUL extrapolation from $D_t$ | **VERIFIED** | `rul_validation.py`: At cycle $t$, only $D_0 \dots D_t$ are passed to polyfit; no future capacity information leaks. | Retain causality claim; document that slope estimation uses only past window $[t-k+1, t]$. | Polynomial slope can be sensitive to window size $k$. |
| **C-17** | Latent-$D_t$ RUL universally outperforms baselines | **FALSE** | `eval_rul_cell_breakdown.py`: Beats RF/XGB on B0005 (4.94 vs 9.20) and B0018 (7.35 vs 20.53), but loses on B0006 (16.99 vs 8.77/8.27). | State: Effective on accelerating-knee trajectory (B0005), but does not uniformly outperform simpler baselines across all cells. | 2 wins, 1 loss vs. tree baselines across valid EOL cells. |
| **C-18** | B0007 right-censoring at 1.40 Ah EOL | **VERIFIED** | `data_loader.py` & `B0007.mat`: Minimum capacity is 1.4005 Ah > 1.40 Ah. Never reaches 70% EOL. | Exclude B0007 from failure-time RUL metrics; retain for macro SoH evaluation. | Right-censored cell cannot validate end-of-life forecasting. |
| **C-19** | Monotonicity ablation causes 203% error explosion | **VERIFIED** | `ablation_table.csv`: Full SLAC = $0.0573 \pm 0.0138$, NoMono = $0.1739 \pm 0.1342$ ($+203.49\%$, 0/12 monotonic). | Report exact percentage ($+203.5\%$); phrase as empirical benchmark finding, not universal physical proof. | Tested on 4 LOBO folds with 3 seeds. |
| **C-20** | Statistically significant superiority ($p < 0.05$) | **FALSE** | Two-sided Wilcoxon signed-rank test has minimum possible $p = 0.125$ ($N=4$) and $p = 0.250$ ($N=3$). | Prohibit claims of formal statistical significance; treat repeated runs as stochastic seeds, not independent cells. | Fundamental small-$N$ benchmark constraint. |
| **C-21** | 32 bytes RAM runs the entire neural network | **FALSE** | `pi_mdcnet_inference.h`: `sizeof(pimdcnet_state_t)` = 32 bytes. ONNX model is ~318 KB; parameters are ~78k float32. | Define 32 bytes strictly as persistent streaming BMS state; distinguish C demo from on-device neural tensor inference. | C code demonstrates state lifecycle, not on-device tensor runtime. |
| **C-22** | ESP32-S3 uses ARM Cortex-M architecture | **FALSE** | ESP32-S3 hardware specification uses dual-core 32-bit Tensilica **Xtensa LX7**. | Correct every instance to "ESP32-S3 / Xtensa LX7". Avoid conflation with RP2040 (ARM Cortex-M0+). | Hardware naming correction. |

---

## 3. Corrected Contribution & Novelty Hierarchy

To ensure publication safety, PI-MDCNet's contributions are structured hierarchically from strongest empirical evidence to engineering deployment scope:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │ TIER 1: ARCHITECTURALLY MONOTONIC CUMULATIVE DAMAGE     │
                  │ ΔD_t = λ · Softplus(f_θ(s_t)) > 0, D_t = D_{t-1} + ΔD_t │
                  │ Guaranteed monotonicity by parameterization (12/12)     │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                  ┌────────────────────────────▼────────────────────────────┐
                  │ TIER 2: EXPLICIT LATENT SOH DECOMPOSITION               │
                  │ SOH_t = SOH_0 - D_t + R_t                               │
                  │ Hard-gated rest relaxation: R_t = rest_flag · R_head    │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                  ┌────────────────────────────▼────────────────────────────┐
                  │ TIER 3: DEDICATED STRESSOR-CONDITIONED DAMAGE PATHWAY   │
                  │ s_t = [mean_temp_cycle, c_rate] directly parameterizes  │
                  │ damage rate, decoupled from rest-signal leakage         │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                  ┌────────────────────────────▼────────────────────────────┐
                  │ TIER 4: CAUSAL DAMAGE-LATENT RUL EXTRAPOLATION          │
                  │ Extrapolation on D_t robust to knee acceleration (B0005)│
                  │ Honest per-cell evaluation (2 wins, 1 loss vs trees)    │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                  ┌────────────────────────────▼────────────────────────────┐
                  │ TIER 5: STREAMING EMBEDDED BMS STATE HARNESS            │
                  │ 32-byte persistent state struct, zero heap allocation   │
                  │ Sequential ONNX parity validated (< 1.19e-07)           │
                  └─────────────────────────────────────────────────────────┘
```

### Primary Contribution (Tier 1)
**Architecturally Constrained Monotonic Cumulative Damage:**
Rather than relying on soft monotonicity loss penalties ($\lambda_{\text{mono}} \text{ReLU}(-\Delta \text{SoH})$) that are routinely violated during unconstrained test inference, PI-MDCNet enforces strict monotonicity on irreversible damage via positive parametric increments:
$$\Delta D_t = \lambda \cdot \text{Softplus}(f_\theta(s_t)), \quad D_t = D_{t-1} + \Delta D_t$$
with $\lambda = 0.003 > 0$. In ablation testing, removing this parametric constraint caused an average SoH RMSE degradation of $+203.5\%$ ($0.0573 \to 0.1739$) and $0/12$ monotonic trajectories, establishing its decisive inductive utility on this benchmark.

### Secondary Contribution (Tier 2)
**Explicit Persistent Degradation and Reversible Relaxation Decomposition:**
Formulating State of Health as:
$$\text{SoH}_t = \text{SoH}_0 - D_t + R_t$$
where cumulative damage $D_t$ is strictly monotonic, and capacity regeneration $R_t$ is hard-gated by the operational rest indicator ($R_t \equiv 0.0$ on active cycles). This prevents the model from predicting unphysical capacity recovery during active discharge while preserving legitimate electrochemical relaxation following rest periods.

### Third Contribution (Tier 3)
**Stressor-to-Damage Routing:**
The operational stressor vector $s_t = [\text{mean\_temp\_cycle}, \text{c\_rate}]$ routes operating severity directly into the damage branch rather than through a shared backbone, preventing rest-related features from corrupting irreversible degradation estimates.

### Fourth Contribution (Tier 4)
**Causal Damage-Latent RUL Extrapolation:**
Extrapolating Remaining Useful Life from the smoothed, monotonic latent damage trajectory $D_t$ rather than noisy, rebound-contaminated capacity traces. On cell B0005, which exhibits an accelerating degradation knee (acceleration ratio $1.67\times$), latent-$D_t$ extrapolation achieves 4.94 cycles MAE, reducing error by ~46% compared to strong tree ensembles (9.20 for Random Forest, 9.21 for XGBoost). However, on cell B0006, which exhibits decelerating degradation (acceleration ratio $0.42\times$), latent-$D_t$ extrapolation yields 16.99 cycles MAE, losing to tree baselines (8.77 for Random Forest, 8.27 for XGBoost).

### Engineering Contribution (Tier 5)
**Streaming BMS Persistent-State Architecture:**
The macro degradation state is compressed into a 32-byte persistent struct (`pimdcnet_state_t`) requiring zero heap allocation. The neural model exports to a 318 KB ONNX binary, verified to achieve numerical parity ($< 1.19 \times 10^{-7}$ maximum difference) across 50 sequential recurrent cycles.

---

## 4. Literature Comparison & Primary-Source Verification

### 4.1 Comparison with Qin et al. (2016, 2019)
* **Citation:** Qin, P., et al., *"A rest-time-based prognostic framework for state-of-health estimation of lithium-ion batteries with capacity regeneration,"* *Energies*, 2016.
* **Their Approach:** Qin et al. explicitly decouple global degradation from local capacity regeneration. They detect regeneration events when adjacent cycle differences exceed a threshold, model regeneration amplitude as an empirical logarithmic/exponential function of rest time, and track global degradation using empirical models (e.g., double exponential) via Particle Filtering (PF).
* **Defensible Distinction:** PI-MDCNet does **not** claim to be the first model to link regeneration to rest time or separate degradation from recovery. The distinction is methodological:
  1. Qin et al. use a statistical filtering framework (Particle Filter) with empirical algebraic models; PI-MDCNet formulates an end-to-end differentiable neural state-space model.
  2. In Qin et al., regeneration events are detected retrospectively via capacity jump thresholds; in PI-MDCNet, regeneration is gated directly by the binary operational rest flag ($R_t = \text{rest\_flag}_t \cdot R_{\text{head}}$).
  3. PI-MDCNet generates cumulative damage dynamically from measured operating stressors ($s_t$) rather than fitting a static aging curve.

### 4.2 Comparison with Monotonic Battery PINNs (2023–2025)
* **Representative Literature:** Recent studies incorporate monotonicity into battery health modeling via Physics-Informed Neural Networks (PINNs) or loss regularization (e.g., adding $\mathcal{L}_{\text{mono}} = \text{ReLU}(\Delta \text{SoH})$ to penalize positive slopes).
* **Defensible Distinction:** PI-MDCNet does **not** claim to be the first monotonic battery model. The technical distinction lies in **parametric vs. regularized enforcement**:
  1. Soft loss penalties encourage monotonicity during training, but do not mathematically prevent non-monotonic outputs at test time on unseen out-of-distribution cells.
  2. By parameterizing incremental damage as $\Delta D_t = \lambda \cdot \text{Softplus}(f_\theta(s_t))$ with $\lambda > 0$, monotonicity is an invariant architectural property ($D_t > D_{t-1}$ strictly holds for all weights and inputs).
  3. Prior monotonic models force the entire SoH to be monotonic, which corrupts real electrochemical relaxation. PI-MDCNet enforces monotonicity **only on the latent damage branch $D_t$**, leaving the composite health $\text{SoH}_t = \text{SoH}_0 - D_t + R_t$ free to capture valid rest rebounds.

### 4.3 Multi-Timescale Battery Prognostics Literature
* **Prior Works:** Multi-timescale models (e.g., dual extended Kalman filters, macro-micro neural networks) have previously separated intra-cycle state estimation from inter-cycle aging.
* **Defensible Distinction:** PI-MDCNet adopts this standard multi-timescale hierarchy as a design choice:
  - Micro-scale: fast intra-cycle SoC tracking via a lightweight GRU on $[V(t), I(t), t]$.
  - Macro-scale: inter-cycle capacity fade and RUL via SLAC on cycle summary features.
  The multi-timescale framing is acknowledged as an established engineering paradigm, not a novel theoretical discovery.

---

## 5. Faculty-Facing Oral Defense Script

### Q1: "What exactly is novel about this work?"
> **Answer:** "We do not claim ground-up novelty on the concepts of battery degradation, rest-period recovery, or monotonicity—all of which have prior literature. Our contribution is an architectural synthesis: we structurally enforce monotonicity on the cumulative damage latent state via a strictly positive parametric formulation ($\Delta D_t = \lambda \cdot \text{Softplus}(f_\theta(s_t))$), while explicitly decoupling it from an independently hard-gated reversible relaxation branch ($R_t$). In our ablation on the NASA benchmark, removing this architectural constraint degrades SoH RMSE by 203.5% and causes 100% of test trajectories to violate monotonicity. That empirical inductive bias is our primary contribution."

### Q2: "Hasn't battery regeneration already been modeled by people like Qin et al.?"
> **Answer:** "Yes, absolutely. Qin et al. (2016) established that capacity regeneration is strongly correlated with rest duration and decoupled it from global degradation using particle filters and empirical regression. We do not claim to have discovered regeneration modeling. Our contribution is embedding that separation into an end-to-end differentiable neural architecture where irreversible degradation is driven by operating stressors and reversible relaxation is hard-gated by the operational rest indicator."

### Q3: "Hasn't monotonicity already been used in battery PINNs?"
> **Answer:** "Yes, monotonicity constraints are common in recent battery literature, but they are almost universally implemented as soft penalty terms in the loss function ($\mathcal{L}_{\text{mono}} = \text{ReLU}(\Delta \text{SoH})$). At test time, soft penalties can be violated when evaluating out-of-distribution cells. Furthermore, enforcing monotonicity on the raw SoH output suppresses legitimate physical recovery after rest. In SLAC, monotonicity is guaranteed architecturally on the latent damage branch, while allowing composite SoH to exhibit valid rebounds."

### Q4: "Why does cross-attention matter if your ablation shows virtually zero difference?"
> **Answer:** "Under this specific benchmark, cross-attention does not materially improve accuracy. Our 4-way ablation shows Full SLAC achieves $0.0573 \pm 0.0138$ SoH RMSE while removing cross-attention achieves $0.0574 \pm 0.0141$—a negligible difference of 0.0001. We explicitly state in our report that cross-attention is an architectural mechanism for multimodal alignment, but its isolated quantitative benefit on this 4-cell dataset is negligible. We do not claim it as a primary performance driver."

### Q5: "Your macro SoH RMSE (0.0573) is worse than Random Forest (0.0300) and XGBoost (0.0306). Why should anyone use your model?"
> **Answer:** "If the sole objective is static interpolation of cycle capacity from summary features, tree ensembles are indeed superior. However, tree ensembles cannot enforce physical monotonicity on unseen cells, do not separate irreversible damage from rest relaxation, and cannot extrapolate causal degradation knees. When extrapolating RUL on the accelerating-knee cell B0005, Random Forest yields an MAE of 9.20 cycles and XGBoost yields 9.21 cycles, whereas our latent damage extrapolation achieves 4.94 cycles. PI-MDCNet provides a physically constrained latent representation designed for streaming BMS integration and causal prognostics, not purely a tabular regression score."

### Q6: "You only have four cells in the dataset. How can you claim statistical superiority?"
> **Answer:** "We cannot and do not claim formal statistical significance. With $N=4$ independent battery cells, the minimum possible two-sided $p$-value for a Wilcoxon signed-rank test is $p = 0.125$; for RUL with $N=3$ valid cells, the minimum possible $p$-value is $p = 0.250$. It is mathematically impossible to achieve $p < 0.05$ on this benchmark. Repeated seeds are stochastic training initializations, not independent physical battery instances. All statistical comparisons in our paper are reported as descriptive and exploratory."

### Q7: "Is your intra-cycle SoC ground truth really ground truth?"
> **Answer:** "No, and we have explicitly corrected that terminology in our documentation. It is not an independent physical state measurement like chemical titration or an embedded reference electrode. It is a retrospective cycle-normalized reference derived from trapezoidal Coulomb integration using the observed total discharge capacity of that specific cycle: $1 - q(t)/Q_{\text{cycle}}$. Crucially, $Q_{\text{cycle}}$ is withheld from the model during inference; the GRU receives only high-frequency voltage, current, and elapsed time."

### Q8: "Does your 32-byte claim mean the entire neural network fits in 32 bytes?"
> **Answer:** "No. The neural network artifact (`pi_mdcnet_edge.onnx`) is approximately 318 KB with ~78,000 float32 parameters. The 32 bytes refers strictly to the persistent streaming BMS state structure (`pimdcnet_state_t`), which holds four fields across cycles: initial SoH (4 bytes), previous damage $D_{t-1}$ (4 bytes), a rolling 5-cycle damage buffer (20 bytes), and a cycle counter (4 bytes). Total: exactly 32 bytes."

### Q9: "Is your embedded demo actually executing the learned neural network?"
> **Answer:** "The pure C code in `embedded/esp32/` is an analytical streaming state harness with representative degradation and relaxation closures designed to validate the 32-byte persistent state lifecycle, timing, and zero-heap execution on microcontrollers like the ESP32-S3 (Xtensa LX7). Full neural tensor inference is executed via ONNX Runtime, which compiles the graph to CPU inside AWS Lambda or an edge gateway."

### Q10: "Why is your RUL model better on B0005 but worse on B0006?"
> **Answer:** "The degradation dynamics of the two cells are fundamentally different. B0005 exhibits an accelerating degradation knee near end-of-life, with a late-to-early degradation slope ratio of $1.67\times$. The monotonic latent damage rate accelerates into the knee, allowing linear extrapolation on $D_t$ to anticipate failure with 4.94 cycles MAE (a ~46% error reduction compared to Random Forest at 9.20 and XGBoost at 9.21). In contrast, B0006 is a decelerating trajectory with an acceleration ratio of $0.42\times$ (late slope is slower than early slope). On decelerating trajectories, our extrapolation overestimates degradation velocity, resulting in 16.99 cycles MAE where simple linear extrapolation (7.73) and tree models (8.27) perform better. We present this openly as an honest limitation: latent damage extrapolation excels on accelerating knees, but not uniformly across all degradation shapes."

### Q11: "Where is the evidence that your model generalizes to other batteries or chemistries?"
> **Answer:** "There is currently no external evidence. The model has been validated exclusively on the 4-cell NASA PCoE dataset (LiCoO2 chemistry). We have not yet validated on Oxford, CALCE, or Stanford datasets. Cross-chemistry and cross-protocol generalization remains unproven future work, and we state this explicitly as a limitation in the paper."

---

## 6. Final Claim Policy (Green / Yellow / Red)

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      FINAL CLAIM POLICY                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

🟢 GREEN — Safe to State Directly (Backed by Mathematical Proof or Code/Data Verification)
───────────────────────────────────────────────────────────────────────────────────────────────────
• Cumulative damage D_t is monotonically non-decreasing by parametric construction (Softplus > 0).
• Removing architectural monotonicity degrades SoH RMSE by +203.5% on this benchmark (0.0573 to 0.1739).
• The model explicitly separates persistent damage (D_t) from rest-gated recovery (R_t).
• R_t is hard-gated to exactly 0.0 on active cycles without preceding rest.
• The SoC target is a retrospective cycle-normalized reference; Q_cycle is strictly withheld at inference.
• Intra-cycle GRU achieves 2.73% overall SoC RMSE on [V, I, t] inputs across 4 LOBO folds.
• Maximum terminal Coulomb-counting error reaches 41.37% on B0006 due to nominal capacity divergence.
• Latent-D RUL extrapolation achieves 4.94 MAE on the accelerating knee of B0005.
• B0007 is right-censored (min capacity 1.4005 Ah > 1.40 Ah EOL threshold).
• NASA cutoff voltages are B0005=2.7V, B0006=2.5V, B0007=2.2V, B0018=2.5V.
• Persistent streaming BMS state footprint is exactly 32 bytes (pimdcnet_state_t) with zero heap allocation.
• ESP32-S3 target processor is the dual-core 32-bit Xtensa LX7.
• ONNX streaming model artifact size is ~318 KB with ~78k parameters.
• Sequential PyTorch vs. ONNX Runtime numerical parity is verified (< 1.19e-07 max absolute diff).

🟡 YELLOW — State Only with Qualification (Context-Dependent / Mixed Evidence)
───────────────────────────────────────────────────────────────────────────────────────────────────
• RUL prognostics: Effective on accelerating knees (B0005: 4.94 vs 9.20 trees), but loses on decelerating
  trajectories (B0006: 16.99 vs 8.27 trees). Must state per-cell win/loss record (2 wins, 1 loss vs trees).
• Cross-attention: Inductive mechanism for fusing electrical/thermal features, but its isolated quantitative
  contribution over simple concatenation is negligible (0.0573 vs 0.0574 RMSE).
• Stressor conditioning: Direct routing of [mean_temp_cycle, c_rate] to damage rate prevents rest leakage,
  but input space is strictly 2D and ambient temperature in dataset is nominal 24°C.
• Embedded deployment: Demonstrated as a pure C streaming state-update engine on 32-byte persistent state;
  full neural tensor graph is compiled via ONNX Runtime for serverless/edge host execution.
• Generalization: Validated across 4 LOBO folds within NASA PCoE; external generalization is unproven.

🔴 RED — Strictly Prohibited (False, Overstated, or Scientifically Unsound)
───────────────────────────────────────────────────────────────────────────────────────────────────
• DO NOT claim: "First ever model to separate battery degradation and recovery." (Preceded by Qin et al.)
• DO NOT claim: "First ever monotonic battery neural network." (Preceded by monotonic PINNs & penalty models.)
• DO NOT claim: "Statistically significant improvement (p < 0.05)." (Mathematically impossible with N=4/N=3.)
• DO NOT claim: "N=12 independent battery experiments." (Pseudoreplication; 3 seeds on 4 physical cells.)
• DO NOT claim: "The entire neural network runs in 32 bytes of RAM." (ONNX model is ~318 KB.)
• DO NOT claim: "Independently measured SoC ground truth." (Reference is retrospective Coulomb integration.)
• DO NOT claim: "Cross-attention is the primary driver of high accuracy." (Isolated gain is 0.0001 RMSE.)
• DO NOT claim: "Universally superior RUL forecasting." (Loses to linear extrapolation and trees on B0006.)
• DO NOT claim: "Validated across multiple battery chemistries and external datasets." (NASA PCoE only.)
• DO NOT claim: "ESP32-S3 is an ARM Cortex-M processor." (ESP32-S3 is Xtensa LX7.)
• DO NOT claim: "Stressor vector s_t contains Depth-of-Discharge (DoD)." (s_t is 2D: temp and C-rate.)
• DO NOT claim: "B0007 discharge cutoff is 2.7 V." (Actual cutoff is 2.2 V.)
```

---

## 7. Experimental Verification Log

### Test Suite Execution
```powershell
.venv\Scripts\pytest -o pythonpath=. tests
```
* **Result:** `12 passed in 38.99s`
* **Coverage:** Baselines (`test_baselines.py`), Data Loader (`test_data_loader.py`), Statistical Evaluation (`test_evaluate.py`), SLAC Architecture (`test_pi_mdcnet.py`).
* **Status:** **PASS**

### Embedded Demo Recompilation & Execution
```powershell
gcc -Wall -Wextra -O2 embedded/esp32/main/main.c embedded/esp32/main/pi_mdcnet_inference.c -o embedded/esp32_demo.exe
.\embedded\esp32_demo.exe
```
* **Output Banner:** `Target Architecture: ESP32-S3 / Xtensa LX7 BMS Controller`
* **Output Memory:** `Initialized BMS State: SoH_0 = 1.000, Persistent State = 32 bytes`
* **Output Footprint:** `Persistent BMS state footprint: 32 bytes static state, zero heap allocation`
* **Status:** **PASS**

### ONNX Export & Multi-Step Sequential Parity
```powershell
.venv\Scripts\python -m embedded.export_onnx
```
* **Model Size:** 318,034 bytes (~318 KB)
* **Single-Step Parity:** Maximum absolute difference across all outputs $\le 5.96 \times 10^{-8}$
* **50-Step Sequential Parity:**
  - `soh`: max diff = $5.96 \times 10^{-8}$
  - `soc`: max diff = $5.96 \times 10^{-8}$
  - `rul`: max diff = $1.19 \times 10^{-7}$
  - `D_t`: max diff = $9.31 \times 10^{-10}$
  - `R_t`: max diff = $1.40 \times 10^{-09}$
  - `D_history_next`: max diff = $9.31 \times 10^{-10}$
* **Tolerance:** All outputs $< 1.00 \times 10^{-5}$
* **Status:** **PASS**

### Naive Coulomb Counting vs. Intra-Cycle GRU Verification
```powershell
.venv\Scripts\python -m experiments.eval_naive_soc
```
* **Naive Coulomb-Counting (Nominal 2.0 Ah):** RMSE = 0.1450 (14.50%), MAE = 0.1127, Max Terminal Error = 41.37% (B0006)
* **Intra-Cycle GRU ([V, I, t]):** RMSE = 0.0273 (2.73%), MAE = 0.0214
* **Per-Cell GRU RMSE:** B0005 = 2.72%, B0006 = 5.13%, B0007 = 1.45%, B0018 = 1.63%
* **Overall Error Reduction:** $-81.2\%$ relative to open-loop Coulomb counting
* **Status:** **PASS**

### RUL Benchmark & Per-Cell Win/Loss Verification
```powershell
.venv\Scripts\python experiments/eval_rul_cell_breakdown.py
```
* **B0005 (Accelerating Knee, Acceleration Ratio 1.67×):**
  - `Latent_Dt_Extrap`: **4.94 ± 0.34 cycles MAE** (WIN vs all)
  - `RandomForest`: 9.20 ± 0.07 cycles MAE
  - `XGBoost`: 9.21 ± 0.84 cycles MAE
  - `Neural_RULHead`: 32.49 ± 0.93 cycles MAE
  - `Stat_Decomp`: 70.49 ± 0.00 cycles MAE
  - `Raw_SoH_Extrap`: 84.60 ± 0.00 cycles MAE
* **B0006 (Decelerating Trajectory, Acceleration Ratio 0.42×):**
  - `Stat_Decomp`: **4.98 ± 0.00 cycles MAE** (WIN)
  - `Raw_SoH_Extrap`: 7.73 ± 0.00 cycles MAE
  - `XGBoost`: 8.27 ± 0.32 cycles MAE
  - `RandomForest`: 8.77 ± 0.05 cycles MAE
  - `Latent_Dt_Extrap`: 16.99 ± 2.60 cycles MAE (LOSS)
  - `Neural_RULHead`: 23.51 ± 0.95 cycles MAE
* **B0018 (Linear Degradation, Acceleration Ratio 0.68×):**
  - `Raw_SoH_Extrap`: **5.80 ± 0.00 cycles MAE** (WIN)
  - `Stat_Decomp`: 6.60 ± 0.00 cycles MAE
  - `Latent_Dt_Extrap`: 7.35 ± 4.10 cycles MAE (Beats trees)
  - `XGBoost`: 20.23 ± 0.25 cycles MAE
  - `RandomForest`: 20.53 ± 0.29 cycles MAE
  - `Neural_RULHead`: 23.35 ± 0.51 cycles MAE
* **B0007:** Right-censored (min capacity 1.4005 Ah > 1.40 Ah; excluded from failure-time metrics).
* **Head-to-Head Win/Loss for Latent-Dt:**
  - vs. `RandomForest`: 2 Wins, 1 Loss (Wins on B0005, B0018; loses on B0006)
  - vs. `XGBoost`: 2 Wins, 1 Loss (Wins on B0005, B0018; loses on B0006)
  - vs. `Neural_RULHead`: 3 Wins, 0 Losses
  - vs. `Raw_SoH_Extrap`: 1 Win, 2 Losses (Decisive win on B0005 knee)
  - vs. `Stat_Decomp`: 1 Win, 2 Losses (Decisive win on B0005 knee)
* **Status:** **PASS**

### 7.5 Canonical Reconciliation: `eval_rul_cell_breakdown.py` vs. `comparison_table.csv`

#### 1. Raw Data Straight from `artifacts/comparison_table.csv` (Off Disk, Unrounded)

```text
           model   fold  seed    rul_mae   rul_rmse
27  RandomForest  B0006    42   8.827022  13.532433
28  RandomForest  B0006    43   8.773892  13.415782
29  RandomForest  B0006    44   8.718071  13.369190
30       XGBoost  B0006    42   8.621956  12.522845
31       XGBoost  B0006    43   8.009653  11.807747
32       XGBoost  B0006    44   8.176432  12.114502
69  RandomForest  B0018    42  20.225496  24.438627
70  RandomForest  B0018    43  20.811824  25.244387
71  RandomForest  B0018    44  20.567422  24.990810
72       XGBoost  B0018    42  20.370591  24.680917
73       XGBoost  B0018    43  20.366960  24.293268
74       XGBoost  B0018    44  19.938331  24.031210
```

And for B0005:
```text
           model   fold  seed    rul_mae   rul_rmse
6   RandomForest  B0005    42   9.191826  12.121400
7   RandomForest  B0005    43   9.131524  12.287541
8   RandomForest  B0005    44   9.271378  12.489001
9        XGBoost  B0005    42   8.633929  13.160493
10       XGBoost  B0005    43   8.831627  13.720503
11       XGBoost  B0005    44  10.171041  15.306619
```

#### 2 & 3. Code Path Investigation & Line Diff

* **The Script that WROTE `comparison_table.csv`:**
  - File: `experiments/train.py`
  - Write Call (Line 597): `results_df.to_csv(output_dir / "comparison_table.csv", index=False)`
  - Evaluation Lines:
    ```python
    # Line 538 in train.py:
    rul_m = compute_metrics(test_df["rul"].values, result.rul, "rul_")

    # Line 148 in train.py (inside compute_metrics):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    mae = float(np.mean(np.abs(yt - yp)))
    ```
    `train.py` evaluates all cycles from cycle $t=0$ to the end of test data without discarding warm-up cycles (`burn_in = 0`).

* **The Script `eval_rul_cell_breakdown.py`:**
  - File: `experiments/eval_rul_cell_breakdown.py`
  - Lines 9–10 & 44:
    ```python
    df_rul = pd.read_csv("artifacts/rul_validation_results.csv")
    df_comp = pd.read_csv("artifacts/comparison_table.csv")
    ...
    v = sub_comp[sub_comp["model"] == tm]["rul_mae"].dropna()
    ```
    `eval_rul_cell_breakdown.py` reads `RandomForest` and `XGBoost` directly from `artifacts/comparison_table.csv`.

* **Direct Truth on 6.85 and 22.72:**
  Neither 6.85 nor 22.72 exists anywhere in this codebase, git history, or generated CSV artifacts. In `comparison_table.csv`, the unrounded values on disk are:
  - B0006: RF = 8.7730, XGBoost = 8.2693
  - B0018: RF = 20.5349, XGBoost = 20.2253
  - B0005: RF = 9.1982, XGBoost = 9.2122

  When `eval_rul_cell_breakdown.py` runs, it outputs these exact values:
  - B0006: RF = 8.77 ± 0.05, XGBoost = 8.27 ± 0.32
  - B0018: RF = 20.53 ± 0.29, XGBoost = 20.23 ± 0.25
  - B0005: RF = 9.20 ± 0.07, XGBoost = 9.21 ± 0.84

#### 4. Single Final Canonical Benchmark Table (One Value Per Cell)

| Cell ID | Latent-$D_t$ Extrap | Random Forest | XGBoost | Neural RUL Head | StatDecomp | Raw SoH Extrap | Latent-$D_t$ vs Trees Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0005** (accelerating knee) | **4.94 ± 0.34** | 9.20 ± 0.07 | 9.21 ± 0.84 | 32.49 ± 0.93 | 70.49 ± 0.00 | 84.60 ± 0.00 | **WIN** (~46% lower MAE than trees) |
| **B0006** (decelerating) | **16.99 ± 2.60** | 8.77 ± 0.05 | 8.27 ± 0.32 | 23.51 ± 0.95 | 4.98 ± 0.00 | 7.73 ± 0.00 | **LOSS** (loses to tree baselines) |
| **B0018** (linear degradation) | **7.35 ± 4.10** | 20.53 ± 0.29 | 20.23 ± 0.25 | 23.35 ± 0.51 | 6.60 ± 0.00 | 5.80 ± 0.00 | **WIN** (~64% lower MAE than trees) |
| **B0007** (right-censored) | *NaN* | *NaN* | *NaN* | *NaN* | *NaN* | *NaN* | *Right-censored (min cap 1.4005 Ah > 1.40 Ah)* |

---

## 8. Final Publication-Safe Contribution Statement

> **Abstract Contribution Summary:**  
> "This work investigates the integration of physical inductive constraints into data-driven battery state estimation and remaining useful life prognostics. We present the Split-Latent Aging Core (SLAC), an architecture that structurally guarantees monotonicity of irreversible battery degradation by parameterizing cumulative damage as a strictly positive increment ($\Delta D_t = \lambda \cdot \text{Softplus}(f_\theta(s_t)) > 0$), while independently modeling reversible electrochemical relaxation through a hard-gated recovery branch ($R_t$). Under Leave-One-Battery-Out cross-validation on the NASA PCoE benchmark, eliminating the architectural monotonicity constraint causes a 203.5% increase in SoH estimation error ($0.0573 \to 0.1739$ RMSE) and results in non-monotonic trajectories across all test folds. Extrapolating remaining useful life from the constrained latent damage trajectory achieves ~46% lower MAE than tree-based models on the accelerating-knee cell (4.94 vs. 9.20 for Random Forest and 9.21 for XGBoost on B0005), though simpler linear models perform better on non-accelerating cells. The macro degradation state is represented in a 32-byte persistent streaming structure requiring zero dynamic memory allocation, suitable for embedded microcontroller tracking. Due to the small sample size ($N=4$ cells), formal statistical significance cannot be claimed ($p \ge 0.125$), and validation on broader chemistries remains essential future work."
