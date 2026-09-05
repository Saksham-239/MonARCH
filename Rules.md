# AI Operating Rules & Project Guardrails (Rules.md)

## 1. Cloud & Financial Safety Guardrails (CRITICAL)

> [!CAUTION]
> The overriding financial requirement of this project is **Zero Unexpected AWS Charges**. The entire cloud footprint must remain within the AWS Free Tier and student credit allotments.

### Strict Service Prohibitions
* **NEVER** provision or write code for:
  - **Amazon Kinesis Data Streams / Firehose:** Incurs continuous hourly shard charges from minute zero.
  - **AWS Glue DPUs:** Billed continuously per DPU-hour.
  - **Amazon Timestream:** No meaningful free tier; charges for memory storage and query processing.
  - **Amazon SageMaker Real-Time Endpoints:** Billed hourly even while idle.
  - **Amazon QuickSight:** Recurring monthly per-user subscription fees.
  - **Amazon QLDB / Managed Kafka (MSK) / OpenSearch Service:** Hourly instance costs.
* **MANDATORY REPLACEMENTS:**
  - Kinesis $\rightarrow$ Direct IoT Core SQL Rule invoking AWS Lambda.
  - Timestream $\rightarrow$ Amazon DynamoDB (25 GB always-free tier, On-Demand billing mode).
  - SageMaker Endpoint $\rightarrow$ Serverless AWS Lambda running quantized ONNX model via `onnxruntime`.
  - QuickSight $\rightarrow$ Static web dashboard hosted on S3 or AWS Amplify free tier.
  - Glue $\rightarrow$ Scheduled Lambda (Python/Pandas) or Step Functions task.
* **AWS Budget Protection:**
  - Deploy an AWS Budget alarm at **$1.00** and **$5.00** thresholds immediately upon cloud initialization.

---

## 2. Machine Learning & Scientific Integrity Rules

### 2.1 Academic Positioning & Honesty
* **NO Claims of Ground-Up Novelty:** Do not claim PI-MDCNet is a brand-new, never-before-seen theoretical architecture. It is an engineering synthesis combining parametric monotonic damage increments ($\Delta D_t = \lambda \cdot \text{Softplus}(f_\theta(s_t))$), hard-gated rest relaxation, and multi-timescale state tracking.
* **NO "First Ever" Claims:** Prior work already separates degradation and regeneration (e.g., Qin et al. 2016) and enforces monotonicity via loss penalties or health indicators. The defensible contribution is the specific architectural enforcement of strictly positive cumulative damage increments rather than relying solely on soft loss penalties.
* **Defensible Contributions Hierarchy:** Frame contributions strictly as:
  1. **Architecturally guaranteed monotonic damage** ($\Delta D_t > 0$ strictly via parameterization).
  2. **Explicit latent decomposition** ($\text{SoH}_t = \text{SoH}_0 - D_t + R_t$) with hard-gated rest relaxation ($R_t = \text{rest\_flag}_t \cdot R_{\text{head}}$).
  3. **Stressor-to-damage routing** ($s_t = [\text{mean\_temp\_cycle}, \text{c\_rate}]$).
  4. **Causal damage-latent RUL extrapolation** (evaluated honestly across valid cells: effective on accelerating-knee cell B0005, but not uniformly superior across all cells).
  5. **Streaming BMS persistent-state design** (32 bytes static state, zero heap allocation).

### 2.2 Electrochemical Fidelity: Relaxation vs. Degradation
* **DO NOT** treat capacity regeneration after long rest periods as "sensor noise" or "an unphysical artifact."
* **DO NOT** apply a blunt single-cycle monotonic clamp across the raw SoH output, because raw capacity physically rebounds after rest.
* **DO** decouple irreversible damage from reversible relaxation:
  - Enforce strict monotonicity on the cumulative damage latent $D_t$.
  - Hard-gate $R_t$ to $0.0$ on all active cycles, allowing non-zero relaxation only on cycles immediately preceded by rest.

### 2.3 Experimental Validation Rigor & Statistical Integrity
* **Leave-One-Battery-Out (LOBO-CV):** Always evaluate models by training on $N-1$ battery cells and testing on the held-out cell across all 4 folds ($B0005, B0006, B0007, B0018$). Chronological within-cell splits are prohibited for cross-cell claims.
* **NO Claims of Formal Statistical Significance:** The benchmark contains only $N=4$ cells (and $N=3$ valid EOL cells for RUL). The minimum possible two-sided Wilcoxon signed-rank $p$-values are $p=0.125$ ($N=4$) and $p=0.250$ ($N=3$). Seeds are repeated stochastic initializations, NOT independent battery samples. Do not treat 12 runs as $N=12$ independent experiments.
* **NO Claim of Cross-Attention Dominance:** The ablation difference between Full SLAC ($0.0573$) and NoCrossAttn ($0.0574$) is negligible ($0.0001$ RMSE). Cross-attention must be described as an architectural mechanism, not a primary performance driver.
* **NO Memory Conflation:** 32 bytes refers strictly to persistent streaming state (`pimdcnet_state_t`). Never claim the neural network (318 KB ONNX) runs in 32 bytes.
* **NO Universal RUL Superiority Claims:** Latent-$D_t$ extrapolation outperforms trees on B0005 and B0018, but is outperformed by linear extrapolation and trees on B0006. All per-cell failure cases must be preserved.
* **NO Unvalidated External Generalization Claims:** The model has been evaluated strictly on the NASA PCoE 4-cell benchmark. External generalization to other chemistries or datasets (Oxford, CALCE, Stanford) remains unproven future work.

---

## 3. Python & Software Engineering Standards

### 3.1 Environment & Dependency Management
* **Never run global `pip install`:** All local code must execute strictly within `.venv`.
* **Explicit paths:** Use `.venv/Scripts/python` and `.venv/Scripts/pip` (on Windows).
* Maintain `requirements.txt` with locked versions.

### 3.2 Code Quality & Reproducibility
* **Deterministic Seeds:** Every script performing training or splitting must set explicit random seeds:
  ```python
  import torch, numpy as np, random
  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)
  ```
* **Type Annotations:** All public functions, classes, and methods must feature Python type hints.
* **Graceful Error Handling:**
  - Wrap network requests, file I/O, and data parsing in structured `try-except` blocks.
  - Log informative error messages rather than failing silently.
  - Sensor telemetry missing values must be interpolated or forward-filled with electrochemical justification.

---

## 4. Cloud Infrastructure as Code (IaC) Rules
* All cloud resources must be defined via **AWS CDK** or **AWS CloudFormation** in the `cloud/` directory.
* Every resource must include cost-tracking tags: `Project: BatteryFleetAnalytics`, `Environment: StudentDev`.
* A single teardown command (`cdk destroy`) must be capable of cleaning up 100% of deployed resources between demo sessions to guarantee zero lingering charges.
