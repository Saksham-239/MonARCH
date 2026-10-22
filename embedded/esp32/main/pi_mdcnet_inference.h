/**
 * @file pi_mdcnet_inference.h
 * @brief PI-MDCNet v4 SLAC Edge Inference Engine for ESP32-S3 (Xtensa LX7) / Embedded BMS
 *
 * Provides a pure C99 streaming execution engine for cycle-by-cycle
 * battery State of Health (SoH), State of Charge (SoC), and Remaining
 * Useful Life (RUL) estimation with split-latent degradation/regeneration
 * decomposition.
 *
 * STRICT ARCHITECTURAL & MEMORY BOUNDARIES:
 * 1. Dependency-free C99: Standard math library (<math.h>) only.
 * 2. Learned Neural Parameters: Exported to static const arrays in Flash ROM (.rodata).
 * 3. Persistent BMS Streaming State: sizeof(pimdcnet_state_t) = exactly 52 bytes
 *    (4B D_prev + 40B D_history[10] + 4B cycle_count + 4B initial_soh_0).
 * 4. Zero Dynamic Heap Allocation: Static arena / stack allocation only; zero malloc/free.
 * 5. Full Real Forward Pass: Real dense linear layers, GELU activations, LayerNorm,
 *    multi-head cross-attention projections, Softplus monotonic damage increment,
 *    hard-gated regeneration branch, and calibrated SoC/RUL heads.
 */

#ifndef PI_MDCNET_INFERENCE_H
#define PI_MDCNET_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#define PIMDCNET_ELEC_DIM     13
#define PIMDCNET_THERM_DIM    4
#define PIMDCNET_STRESSOR_DIM 2
#define PIMDCNET_REST_DIM     3
#define PIMDCNET_RUL_WINDOW_K 10

/**
 * @brief Input telemetry features for a single battery discharge cycle.
 */
typedef struct {
    float x_elec[PIMDCNET_ELEC_DIM];        /**< Electrical summary stats (13) */
    float x_therm[PIMDCNET_THERM_DIM];      /**< Thermal stats (mean, max, std, rise) (4) */
    float s_t[PIMDCNET_STRESSOR_DIM];       /**< Stressors: mean_temp, c_rate (2) */
    float rest_features[PIMDCNET_REST_DIM]; /**< Rest: duration (hrs), onset SoC, ambient temp (3) */
    float rest_flag;                        /**< Binary flag: 1.0 if rest >= threshold, else 0.0 */
    float cycle_idx;                        /**< Normalized cycle index t / 1000.0 (fixed-horizon scale) */
} pimdcnet_cycle_input_t;

/**
 * @brief Persistent state tracked across cycles by the embedded BMS.
 * Total memory: exactly 52 bytes (4B D_prev + 40B D_history + 4B cycle_count + 4B initial_soh_0).
 */
typedef struct {
    float D_prev;                                  /**< Cumulative irreversible damage D_{t-1} */
    float D_history[PIMDCNET_RUL_WINDOW_K];       /**< Rolling buffer of last k damage values */
    uint32_t cycle_count;                         /**< Monotonically increasing cycle counter */
    float initial_soh_0;                          /**< Calibrated initial SoH (typically ~1.0) */
} pimdcnet_state_t;

/**
 * @brief Inference output from PI-MDCNet.
 */
typedef struct {
    float soh;              /**< Predicted State of Health (SoH_0 - D_t + R_t) */
    float soc;              /**< Predicted State of Charge */
    float rul;              /**< Predicted Remaining Useful Life in cycles */
    float D_t;              /**< Updated cumulative irreversible damage (monotonic) */
    float R_t;              /**< Instantaneous capacity regeneration (hard-gated at rest) */
    float delta_D;          /**< Incremental damage added this cycle */
    uint32_t inference_us;  /**< Inference execution latency in microseconds */
} pimdcnet_output_t;

/**
 * @brief Initialize the embedded PI-MDCNet runtime and persistent BMS state.
 *
 * @param state Pointer to state structure to initialize.
 * @param nominal_soh Initial nominal SoH (e.g. 1.0f).
 */
void pimdcnet_init(pimdcnet_state_t *state, float nominal_soh);

/**
 * @brief Reset the internal BMS state for a fresh battery pack.
 *
 * @param state Pointer to state structure.
 */
void pimdcnet_reset(pimdcnet_state_t *state);

/**
 * @brief Execute single-cycle streaming inference with zero heap allocation.
 *
 * Evaluates the complete Split-Latent Aging Core (SLAC):
 * 1. Encodes electrical features: Linear(13->64) -> GELU -> Linear(64->32) -> LayerNorm -> e_t
 * 2. Encodes thermal features: Linear(4->32) -> GELU -> Linear(32->32) -> LayerNorm -> th_t
 * 3. Cross-attention fusion: Projections + Multihead Cross-Attention + LayerNorm -> h_t (64)
 * 4. Damage increment: Linear(2->32) -> GELU -> Linear(32->32) -> GELU -> Linear(32->1)
 *    delta_D = damage_scale * Softplus(raw_damage)
 *    D_t = D_prev + delta_D (strictly monotonic by construction)
 * 5. Reversible regeneration: Rest MLP(3->32->64) -> Cross-Attention with h_t -> MLP(64->32->1) -> Tanh
 *    R_t = rest_flag * r_max * Tanh(MLP) (hard gated to 0.0 when rest_flag == 0)
 * 6. SoH decomposition: SoH = SoH_0 - D_t + R_t
 * 7. SoC head: Linear(64->32) -> GELU -> Linear(32->1) -> Sigmoid
 * 8. RUL head: Rolling history shift + cycle index -> Linear(11->32) -> GELU -> Linear(32->32) -> GELU -> Linear(32->1) -> Softplus
 *
 * @param input Pointer to cycle telemetry.
 * @param state In/Out pointer to persistent BMS state.
 * @param output Pointer to write output predictions.
 * @return 0 on success, negative error code on failure.
 */
int pimdcnet_infer_step(
    const pimdcnet_cycle_input_t *input,
    pimdcnet_state_t *state,
    pimdcnet_output_t *output
);

/**
 * @brief Static profiling metrics for embedded resource accounting.
 */
uint32_t pimdcnet_get_param_count(void);
uint32_t pimdcnet_get_mac_count(void);
uint32_t pimdcnet_get_flash_bytes(void);
uint32_t pimdcnet_get_sram_bytes(void);

#ifdef __cplusplus
}
#endif

#endif /* PI_MDCNET_INFERENCE_H */
