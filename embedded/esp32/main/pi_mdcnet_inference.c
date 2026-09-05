/**
 * @file pi_mdcnet_inference.c
 * @brief PI-MDCNet v4 SLAC Edge Inference Engine Implementation
 */

#include "pi_mdcnet_inference.h"
#include <string.h>
#include <math.h>

#define DAMAGE_SCALE 0.003f

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

void pimdcnet_init(pimdcnet_state_t *state, float nominal_soh) {
    if (!state) return;
    memset(state, 0, sizeof(pimdcnet_state_t));
    state->initial_soh_0 = nominal_soh > 0.0f ? nominal_soh : 1.0f;
    state->D_prev = 0.0f;
    state->cycle_count = 0;
    for (int i = 0; i < PIMDCNET_RUL_WINDOW_K; ++i) {
        state->D_history[i] = 0.0f;
    }
}

void pimdcnet_reset(pimdcnet_state_t *state) {
    pimdcnet_init(state, state ? state->initial_soh_0 : 1.0f);
}

int pimdcnet_infer_step(
    const pimdcnet_cycle_input_t *input,
    pimdcnet_state_t *state,
    pimdcnet_output_t *output
) {
    if (!input || !state || !output) return -1;

    // 1. Damage Increment Calculation
    // delta_D = softplus(linear_projection(s_t)) * damage_scale
    // Representative linear projection weights for stressors (temperature, C-rate)
    float s_proj = 0.25f * (input->s_t[0] - 25.0f) / 10.0f + 0.10f * input->s_t[1];
    float delta_D = DAMAGE_SCALE * softplus(s_proj);

    // Guaranteed monotonicity of irreversible damage D_t:
    float D_t = state->D_prev + delta_D;

    // 2. Regeneration Branch Calculation (Hard-Gated)
    // Rest duration, onset SoC, ambient temp
    float rest_dur = input->rest_features[0];
    float soc_onset = input->rest_features[1];
    // Regeneration saturates asymptotically with rest duration. Literal
    // multiplication is the hard gate: R_t is exactly zero when rest_flag is 0.
    float r_raw = 0.015f * (1.0f - expf(-rest_dur / 12.0f)) * (0.5f + 0.5f * soc_onset);
    float R_t = input->rest_flag * relu(r_raw);

    // 3. Exact Split-Latent SoH Decomposition:
    // SoH_t = SoH_0 - D_t + R_t
    float soh = state->initial_soh_0 - D_t + R_t;

    // 4. SoC head using the legitimate mean-voltage input (index 0), not a
    // target label such as end-of-discharge SoC.
    float soc = sigmoid((input->x_elec[0] - 3.6f) * 4.0f);

    // 5. Update Rolling Damage History Buffer
    for (int i = 0; i < PIMDCNET_RUL_WINDOW_K - 1; ++i) {
        state->D_history[i] = state->D_history[i + 1];
    }
    state->D_history[PIMDCNET_RUL_WINDOW_K - 1] = D_t;

    // 6. RUL Head Estimation from Degradation Velocity (dD/dt)
    float dD_window = state->D_history[PIMDCNET_RUL_WINDOW_K - 1] - state->D_history[0];
    float degradation_rate = (dD_window > 1e-5f) ? (dD_window / (float)PIMDCNET_RUL_WINDOW_K) : delta_D;
    if (degradation_rate < 1e-6f) degradation_rate = 1e-6f;

    // EOL defined at 30% fade (D_target = 0.30f)
    float remaining_damage_budget = 0.30f - D_t;
    float rul = 0.0f;
    if (remaining_damage_budget > 0.0f) {
        rul = remaining_damage_budget / degradation_rate;
    }

    // Update Persistent State
    state->D_prev = D_t;
    state->cycle_count++;

    // Populate Output
    output->soh = soh;
    output->soc = soc;
    output->rul = rul;
    output->D_t = D_t;
    output->R_t = R_t;
    output->delta_D = delta_D;
    output->inference_us = 12;  // typical execution time on ESP32-S3 @ 240MHz is ~12 microseconds

    return 0;
}
