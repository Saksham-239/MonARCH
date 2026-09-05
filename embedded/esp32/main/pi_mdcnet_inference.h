/**
 * @file pi_mdcnet_inference.h
 * @brief PI-MDCNet v4 SLAC Edge Inference Engine for ESP32-S3 (Xtensa LX7) / Embedded BMS
 *
 * Provides a pure C99 streaming execution harness for cycle-by-cycle
 * battery State of Health (SoH), State of Charge (SoC), and Remaining
 * Useful Life (RUL) estimation with split-latent degradation/regeneration
 * decomposition.
 *
 * STRICT ARCHITECTURAL & MEMORY BOUNDARIES:
 * 1. Neural Model Artifact: pi_mdcnet_edge.onnx (~318 KB ONNX binary, opset 14).
 * 2. Learned Neural Parameters: ~78,000 float32 weights (~312 KB raw parameters).
 * 3. Persistent BMS Streaming State: sizeof(pimdcnet_state_t) = exactly 32 bytes
 *    (4B soh_0 + 4B D_prev + 20B D_history[5] + 4B cycle_count).
 * 4. Zero Heap Allocation: Static allocation only; no malloc/free during inference.
 * 5. Demonstration Engine Scope: This pure C implementation is an analytical
 *    streaming state-update demonstration of the 32-byte persistent state lifecycle
 *    and degradation/regeneration closures; full neural tensor graph execution
 *    is performed via ONNX Runtime on serverless/edge host platforms.
 * 6. Executable Memory != Persistent State: ROM/flash executable and temporary stack
 *    frames are distinct from the 32-byte persistent streaming state.
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
#define PIMDCNET_RUL_WINDOW_K 5

/**
 * @brief Input telemetry features for a single battery discharge cycle.
 */
typedef struct {
    float x_elec[PIMDCNET_ELEC_DIM];        /**< Electrical summary stats + ECM fit + dQ/dV */
    float x_therm[PIMDCNET_THERM_DIM];      /**< Thermal stats (mean, max, std, rise) */
    float s_t[PIMDCNET_STRESSOR_DIM];       /**< Stressors: mean_temp, c_rate */
    float rest_features[PIMDCNET_REST_DIM]; /**< Rest: duration (hrs), onset SoC, ambient temp */
    float rest_flag;                        /**< Binary flag: 1.0 if rest >= threshold, else 0.0 */
    float cycle_idx;                        /**< Normalized cycle index t / T_max */
} pimdcnet_cycle_input_t;

/**
 * @brief Persistent state tracked across cycles by the embedded BMS.
 */
typedef struct {
    float D_prev;                                  /**< Cumulative irreversible damage D_{t-1} */
    float D_history[PIMDCNET_RUL_WINDOW_K];       /**< Rolling buffer of last k damage values */
    uint32_t cycle_count;                         /**< Monotonically increasing cycle counter */
    float initial_soh_0;                          /**< Calibrated initial SoH (typically 1.0) */
} pimdcnet_state_t;

/**
 * @brief Inference output from PI-MDCNet.
 */
typedef struct {
    float soh;              /**< Predicted State of Health (SoH_0 - D_t + R_t) */
    float soc;              /**< Predicted end-of-discharge State of Charge */
    float rul;              /**< Predicted Remaining Useful Life in cycles */
    float D_t;              /**< Updated cumulative irreversible damage (monotonic) */
    float R_t;              /**< Instantaneous capacity regeneration (hard-gated at rest) */
    float delta_D;          /**< Incremental damage added this cycle */
    uint32_t inference_us;  /**< Inference execution latency in microseconds */
} pimdcnet_output_t;

/**
 * @brief Initialize the embedded PI-MDCNet runtime and state.
 *
 * @param state Pointer to state structure to initialize.
 * @param nominal_soh Initial nominal SoH (e.g. 1.0f).
 */
void pimdcnet_init(pimdcnet_state_t *state, float nominal_soh);

/**
 * @brief Reset the internal BMS state for a new battery pack.
 *
 * @param state Pointer to state structure.
 */
void pimdcnet_reset(pimdcnet_state_t *state);

/**
 * @brief Execute single-cycle streaming inference.
 *
 * Evaluates the Split-Latent Aging Core (SLAC):
 * 1. Encodes electrical and thermal telemetry
 * 2. Applies cross-attention fusion
 * 3. Computes incremental damage: delta_D = softplus(W*s_t + b) * damage_scale
 * 4. Updates D_t = D_prev + delta_D (architecturally monotonic)
 * 5. Computes hard-gated regeneration: R_t = rest_flag * R_raw
 * 6. Decomposes SoH = SoH_0 - D_t + R_t
 * 7. Predicts RUL from rolling D_t history
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

#ifdef __cplusplus
}
#endif

#endif /* PI_MDCNET_INFERENCE_H */
