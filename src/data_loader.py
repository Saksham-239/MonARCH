"""
PI-MDCNet v4 — SLAC Data Pipeline
===================================
NASA PCoE Li-ion Battery Dataset ingestion, feature extraction, rest-event
metadata computation, and Leave-One-Battery-Out (LOBO) fold construction.

Source: NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset
Cells: B0005, B0006, B0007, B0018 (all rated 2.0 Ah, run-to-failure under
different operational profiles with varying rest schedules).

.mat file schema (documented layout — field names verified at load time):
    cycle[]                      # top-level array of operations
      .type                      # 'charge' | 'discharge' | 'impedance'
      .ambient_temperature       # °C
      .time                      # 6-element MATLAB date vector [year month day hour min sec]
      .data
        .Voltage_measured        # V (discharge)
        .Current_measured        # A (discharge)
        .Temperature_measured    # °C (discharge)
        .Time                    # seconds, per-sample within cycle
        .Capacity                # Ah — ground-truth capacity at this cycle

End-of-life convention: 30% fade from rated capacity (2.0 Ah → 1.4 Ah).
Configurable via EOL_FADE_FRACTION.

References:
    - Saha & Goebel, NASA PCoE dataset documentation
    - Orchard, Saha & Goebel, IEEE Trans. I&M 2013
    - Qin et al. 2016/2019 — rest-time-based prognostic framework (RTPF)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ═══════════════════════════════════════════════════════════════════════════════
# LOCKED CONSTANTS — change only via config, never inline magic numbers
# ═══════════════════════════════════════════════════════════════════════════════

RATED_CAPACITY: float = 2.0
"""Rated (nameplate) capacity in Ah for NASA PCoE cells B0005/B0006/B0007/B0018."""

EOL_FADE_FRACTION: float = 0.30
"""End-of-life defined as this fraction of capacity fade from rated.
   EOL capacity = RATED_CAPACITY * (1 - EOL_FADE_FRACTION) = 1.4 Ah.
   Some papers use 0.20 — this is configurable so both can be reported."""

EOL_CAPACITY: float = RATED_CAPACITY * (1.0 - EOL_FADE_FRACTION)
"""Derived: absolute capacity (Ah) at end-of-life threshold."""

REST_THRESHOLD_HOURS: float = 1.0
"""Minimum gap duration (hours) between consecutive cycles for a cycle to be
   flagged as preceded by a meaningful rest event. Start at 1.0 based on
   typical relaxation literature timescales. Sweep {0.5, 1, 2, 4} in ablation
   since the exact threshold varies by cell chemistry and hasn't been
   independently verified for this specific dataset."""

SAVGOL_WINDOW_LENGTH: int = 21
"""Savitzky-Golay filter window length (samples) for dQ/dV smoothing.
   Must be odd. Chosen to smooth 10 Hz noise without obliterating the
   underlying dQ/dV peak structure. The NASA PCoE discharge curves are
   typically 1000–3000 samples long, so 21 samples ≈ 2s of data."""

SAVGOL_POLY_ORDER: int = 3
"""Savitzky-Golay polynomial order for dQ/dV smoothing.
   Order 3 (cubic) balances noise rejection with preservation of the
   asymmetric dQ/dV peak shape. Order 2 over-smooths peaks; order 4+
   lets through too much high-frequency noise."""

ECM_FIT_MIN_SAMPLES: int = 20
"""Minimum number of samples in a voltage relaxation/discharge segment
   required to attempt a 1RC Thevenin ECM curve fit. Below this, the
   fit is unreliable and we fall back to NaN + interpolation."""

CELL_IDS: list[str] = ["B0005", "B0006", "B0007", "B0018"]
"""NASA PCoE cells used in this study. All are 18650-format commercial
   Li-ion cells aged under charge/discharge cycling with different
   operational profiles (constant-current discharge at different C-rates,
   with varying rest intervals between cycles)."""

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CycleData:
    """Raw extracted data from a single discharge cycle."""
    cycle_index: int
    voltage: NDArray[np.float64]        # V, per-sample
    current: NDArray[np.float64]        # A, per-sample (negative = discharge)
    temperature: NDArray[np.float64]    # °C, per-sample
    time_seconds: NDArray[np.float64]   # seconds, per-sample within cycle
    capacity_ah: float                  # Ah, ground-truth measured capacity
    ambient_temp: float                 # °C, ambient temperature
    cycle_start_timestamp: float         # Unix timestamp (seconds) of cycle start


@dataclass
class CycleFeatures:
    """Engineered features for a single discharge cycle.

    All features are scalar per-cycle values (one row in the output DataFrame),
    except dq_dv_spectrum which is a fixed-length vector of dQ/dV evaluated
    at uniformly spaced voltage bins.
    """
    cycle_index: int

    # ── Labels ──
    soh: float
    """State of Health: Capacity_t / RATED_CAPACITY. Dimensionless, ∈ (0, 1]."""

    soc_eod: float
    """State of Charge at end of discharge: 0.0 by definition for a full
       discharge, but partial discharges may end higher. Computed via Coulomb
       counting against measured (aged) capacity, not nameplate."""

    rul: float
    """Remaining Useful Life in cycles: t_EOL - t, where t_EOL is the cycle
       at which measured capacity crosses EOL_CAPACITY. NaN if this cell
       never reaches EOL in the dataset (right-censored)."""

    # ── Electrical summary stats ──
    v_mean: float
    v_min: float
    v_max: float
    v_std: float
    v_slope: float
    """Linear slope of voltage vs. time over the discharge — proxy for
       internal resistance increase."""

    i_mean: float
    i_std: float

    # ── dQ/dV features ──
    dq_dv_spectrum: NDArray[np.float64]
    """dQ/dV evaluated at DQ_DV_VOLTAGE_BINS uniformly spaced voltage points.
       Smoothed via Savitzky-Golay (window=SAVGOL_WINDOW_LENGTH, order=
       SAVGOL_POLY_ORDER) before differentiation to suppress 10 Hz sampling
       noise. Shape: (N_DQ_DV_BINS,)."""

    dq_dv_peak_v: float
    """Voltage at which dQ/dV reaches its maximum — tracks redox peak shift."""

    dq_dv_peak_mag: float
    """Magnitude of the dQ/dV peak — tracks active material loss."""

    # ── 1RC Thevenin ECM parameters ──
    v_ocv: float
    """Open-circuit voltage estimate (V). Estimated from the relaxed voltage
       at the start of discharge (first few samples after current onset), or
       from an OCV-SoC lookup if relaxation data is insufficient.
       One value per cycle, not per sample."""

    r0: float
    """Ohmic resistance (Ω). Estimated from the instantaneous voltage drop
       at current onset: R0 = ΔV_instant / I_step. One value per cycle."""

    r1: float
    """Polarization resistance (Ω) of the 1RC parallel branch. Fitted via
       nonlinear least squares against the voltage transient after the
       initial ohmic drop. Fit requires ≥ ECM_FIT_MIN_SAMPLES samples in
       the transient region; NaN if insufficient data."""

    c1: float
    """Polarization capacitance (F) of the 1RC parallel branch. Derived
       from the time constant τ₁ = R1 * C1 of the exponential voltage
       relaxation. Same fit constraints as R1."""

    # ── Thermal summary stats ──
    t_mean: float
    t_max: float
    t_std: float
    t_rise: float
    """Temperature rise over the discharge: T_end - T_start (°C)."""

    # ── Degradation stressor vector s_t (v4: decoupled from backbone) ──
    throughput_cycle_ah: float
    """Per-cycle incremental Ah-throughput: ∫|I|dt over THIS cycle only.
       NOT cumulative across cycles. This is deliberately named
       throughput_cycle_ah (not throughput_t) to prevent confusion with
       cumulative throughput. Computed by numerical integration (trapezoidal)
       of |current| over the cycle's time vector.

       This feeds the damage branch's stressor vector s_t directly from raw
       electrical measurements — it is NOT derived from the cross-attention
       backbone (§2.1), so it cannot carry rest-related signal into the
       damage branch (see §2.2 identifiability rationale)."""

    mean_temp_cycle: float
    """Mean measured temperature (°C) during this discharge cycle.
       Part of the stressor vector s_t. Computed from raw per-sample
       temperature readings, not from the thermal encoder output."""

    c_rate: float
    """Discharge C-rate: |I_mean| / RATED_CAPACITY. Dimensionless.
       Part of the stressor vector s_t. Uses rated (not aged) capacity
       in the denominator for consistency across aging states."""

    # ── Rest-event metadata (§1.2) ──
    rest_duration_hours: float
    """Duration of the rest period PRECEDING this cycle, in hours.
       Computed as the gap between this cycle's start time and the end
       of the previous cycle (previous cycle start + duration).
       First cycle of each cell gets rest_duration = 0.0.

       'Rest' here means the absence of any charge/discharge/impedance
       operation — it is the gap between consecutive operations in the
       cycle array, not a separate cycle type in the raw schema."""

    soc_at_rest_onset: float
    """State of Charge at the moment rest began — i.e., SoC at the end
       of the previous cycle. Computed via Coulomb counting against
       the previous cycle's measured capacity. First cycle gets NaN."""

    ambient_temp_rest: float
    """Ambient temperature during the rest period (°C). Approximated as
       the average of the previous cycle's and this cycle's ambient_temperature
       fields. First cycle uses this cycle's ambient_temperature only."""

    rest_flag: int
    """Binary flag: 1 if rest_duration_hours > REST_THRESHOLD_HOURS, else 0.
       This is the hard gate for the regeneration branch (§2.3) — the model
       is architecturally incapable of using R_t on cycles where this is 0."""


# ═══════════════════════════════════════════════════════════════════════════════
# dQ/dV VOLTAGE BINS
# ═══════════════════════════════════════════════════════════════════════════════

DQ_DV_VOLTAGE_MIN: float = 2.7
"""Lower voltage bound (V) for dQ/dV spectrum binning. Below this, the
   discharge curve is in the steep cutoff region and dQ/dV is dominated
   by noise / numerical artifacts."""

DQ_DV_VOLTAGE_MAX: float = 4.2
"""Upper voltage bound (V) for dQ/dV spectrum binning. Above this, the cell
   is at or near full charge — the interesting dQ/dV peaks live in the
   plateau region between 3.0–3.8 V for these NMC/LCO chemistry cells."""

N_DQ_DV_BINS: int = 50
"""Number of uniformly spaced voltage bins for the dQ/dV spectrum.
   50 bins over [2.7, 4.2] V → 30 mV resolution — sufficient to resolve
   the main redox peaks without overfitting to noise."""

DQ_DV_VOLTAGE_BINS: NDArray[np.float64] = np.linspace(
    DQ_DV_VOLTAGE_MIN, DQ_DV_VOLTAGE_MAX, N_DQ_DV_BINS
)


# ═══════════════════════════════════════════════════════════════════════════════
# RAW .mat FILE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _datevec_to_timestamp(datevec: NDArray) -> float:
    """Convert a MATLAB date vector to a Unix-like timestamp (seconds).

    The NASA PCoE .mat files store cycle start times as 6-element MATLAB
    date vectors: [year, month, day, hour, minute, second].
    This is NOT a datenum (days since epoch) — it's a structured datetime.

    We convert to seconds since an arbitrary epoch for computing durations
    between cycles. The absolute value doesn't matter — only differences.
    """
    from datetime import datetime

    dv = datevec.flatten()
    if len(dv) < 6:
        return np.nan

    try:
        dt = datetime(
            int(dv[0]), int(dv[1]), int(dv[2]),
            int(dv[3]), int(dv[4]), int(dv[5]),
        )
        return dt.timestamp()
    except (ValueError, OverflowError):
        return np.nan


def _load_mat_hdf5(filepath: Path) -> list[dict[str, Any]]:
    """Load a NASA PCoE .mat file in HDF5 (MATLAB v7.3) format.

    The NASA PCoE dataset distributes .mat files that may be either MATLAB v5
    or v7.3 (HDF5) format depending on the download mirror. This function
    handles the HDF5 variant.

    Parameters
    ----------
    filepath : Path
        Path to the .mat file (e.g., B0005.mat).

    Returns
    -------
    list[dict]
        List of cycle dicts with keys: 'type', 'ambient_temperature', 'time',
        'data' (sub-dict with voltage/current/temperature/time/capacity arrays).

    Raises
    ------
    FileNotFoundError
        If the .mat file does not exist.
    KeyError
        If expected fields are missing from the .mat structure.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"NASA PCoE .mat file not found: {filepath}")

    cycles: list[dict[str, Any]] = []

    with h5py.File(filepath, "r") as f:
        # Navigate HDF5 structure: top-level key is the cell name (e.g. 'B0005')
        cell_name = filepath.stem
        # The structure may be nested differently — try common layouts
        if cell_name in f:
            root = f[cell_name]
        else:
            # Some mirrors have the data at the root level
            root = f

        # Access the 'cycle' field
        if "cycle" not in root:
            raise KeyError(
                f"Expected 'cycle' field in {filepath}. "
                f"Available keys: {list(root.keys())}"
            )

        cycle_group = root["cycle"]

        # HDF5 stores struct arrays as groups of datasets
        # Each field (type, ambient_temperature, etc.) is an array of refs or values
        n_cycles = cycle_group["type"].shape[0]

        for i in range(n_cycles):
            try:
                # Dereference type string
                type_ref = cycle_group["type"][i, 0]
                type_str = _deref_string(f, type_ref)

                # Ambient temperature
                amb_ref = cycle_group["ambient_temperature"][i, 0]
                ambient_temp = float(np.array(f[amb_ref]).flatten()[0])

                # Cycle start time (MATLAB datenum)
                time_ref = cycle_group["time"][i, 0]
                cycle_time = float(np.array(f[time_ref]).flatten()[0])

                # Data sub-struct
                data_ref = cycle_group["data"][i, 0]
                data_group = f[data_ref]

                data_dict = {}
                for key in data_group.keys():
                    arr = np.array(data_group[key]).flatten().astype(np.float64)
                    data_dict[key] = arr

                cycles.append({
                    "type": type_str,
                    "ambient_temperature": ambient_temp,
                    "time": cycle_time,
                    "data": data_dict,
                })
            except Exception as e:
                logger.warning(
                    "Skipping cycle %d in %s: %s", i, filepath.name, e
                )
                continue

    logger.info("Loaded %d cycles from %s (HDF5 format)", len(cycles), filepath.name)
    return cycles


def _deref_string(f: h5py.File, ref: Any) -> str:
    """Dereference an HDF5 object reference to a Python string."""
    obj = f[ref]
    raw = np.array(obj).flatten()
    if raw.dtype.kind in ("U", "S", "O"):
        return str(raw[0])
    # Character array stored as uint16
    return "".join(chr(int(c)) for c in raw)


def _load_mat_v5(filepath: Path) -> list[dict[str, Any]]:
    """Load a NASA PCoE .mat file in MATLAB v5 format.

    Uses scipy.io.loadmat with struct_as_record=True to get numpy record
    arrays. The actual NASA PCoE structure is:
        mat['B0005'][0,0]['cycle'][0, i] → record with fields:
            'type', 'ambient_temperature', 'time', 'data'
        time is a 6-element date vector [year, month, day, hour, min, sec]
        data is a sub-record with Voltage_measured, Current_measured, etc.

    Parameters
    ----------
    filepath : Path
        Path to the .mat file.

    Returns
    -------
    list[dict]
        Same format as _load_mat_hdf5.
    """
    import scipy.io as sio

    if not filepath.exists():
        raise FileNotFoundError(f"NASA PCoE .mat file not found: {filepath}")

    mat = sio.loadmat(str(filepath), struct_as_record=True)
    cell_name = filepath.stem

    if cell_name not in mat:
        struct_keys = [k for k in mat.keys() if not k.startswith("__")]
        if len(struct_keys) == 1:
            cell_name = struct_keys[0]
        else:
            raise KeyError(
                f"Cannot identify cell struct in {filepath}. "
                f"Keys: {struct_keys}"
            )

    # Navigate: mat[cell_name] is shape (1,1), field 'cycle' is shape (1, N)
    cell_struct = mat[cell_name]
    cycle_arr = cell_struct[0, 0]['cycle']
    n_cycles = cycle_arr.shape[1]

    cycles: list[dict[str, Any]] = []

    for i in range(n_cycles):
        try:
            cyc = cycle_arr[0, i]

            # Type string
            type_raw = cyc['type']
            type_str = str(type_raw.flatten()[0]).strip()

            # Ambient temperature
            amb_raw = cyc['ambient_temperature']
            ambient_temp = float(amb_raw.flatten()[0])

            # Time: 6-element MATLAB date vector [year month day hour min sec]
            time_raw = cyc['time']
            time_vec = time_raw.flatten()
            cycle_timestamp = _datevec_to_timestamp(time_vec)

            # Data sub-struct: shape (1,1)
            data_struct = cyc['data']
            if data_struct.size == 0:
                logger.warning("Cycle %d in %s has empty data, skipping", i, filepath.name)
                continue

            data_rec = data_struct[0, 0]
            data_dict: dict[str, NDArray[np.float64]] = {}

            for field_name in data_rec.dtype.names:
                val = data_rec[field_name]
                # Impedance cycles contain complex-valued data (Z_real + Z_imag).
                # We only use discharge cycles, so safely take the real part.
                raw = np.asarray(val).flatten()
                if np.iscomplexobj(raw):
                    raw = raw.real
                arr = raw.astype(np.float64)
                if arr.size > 0:
                    data_dict[field_name] = arr

            if not data_dict:
                logger.warning("Cycle %d in %s has no data fields, skipping", i, filepath.name)
                continue

            cycles.append({
                "type": type_str,
                "ambient_temperature": ambient_temp,
                "time": cycle_timestamp,  # seconds since epoch
                "data": data_dict,
            })
        except Exception as e:
            logger.warning("Skipping cycle %d in %s: %s", i, filepath.name, e)
            continue

    logger.info("Loaded %d cycles from %s (v5 format)", len(cycles), filepath.name)
    return cycles


def load_mat_file(filepath: Path) -> list[dict[str, Any]]:
    """Load a NASA PCoE .mat file, auto-detecting v5 vs v7.3 (HDF5) format.

    Parameters
    ----------
    filepath : Path
        Path to the .mat file.

    Returns
    -------
    list[dict]
        List of cycle dicts.
    """
    # HDF5 files start with the HDF5 magic bytes
    with open(filepath, "rb") as fh:
        magic = fh.read(4)

    if magic == b"\x89HDF":
        return _load_mat_hdf5(filepath)
    else:
        return _load_mat_v5(filepath)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_discharge_cycles(
    all_cycles: list[dict[str, Any]],
) -> list[CycleData]:
    """Filter to discharge cycles and extract raw CycleData structs.

    Only discharge cycles are used for SoH/RUL labels. Charge and impedance
    cycles are skipped but their timestamps are preserved for rest-duration
    computation (handled separately in _compute_rest_metadata).

    Parameters
    ----------
    all_cycles : list[dict]
        Raw cycles from load_mat_file.

    Returns
    -------
    list[CycleData]
        Ordered list of discharge CycleData, with cycle_index corresponding
        to the discharge cycle's position in the original cycle array.
    """
    discharge_cycles: list[CycleData] = []

    for i, cyc in enumerate(all_cycles):
        if cyc["type"].lower() != "discharge":
            continue

        data = cyc["data"]

        # Identify voltage/current field names (vary between charge/discharge)
        v_key = _find_key(data, ["Voltage_measured", "Voltage_load"])
        i_key = _find_key(data, ["Current_measured", "Current_load"])
        t_key = _find_key(data, ["Temperature_measured"])
        time_key = _find_key(data, ["Time"])
        cap_key = _find_key(data, ["Capacity"])

        if any(k is None for k in [v_key, i_key, t_key, time_key, cap_key]):
            logger.warning(
                "Discharge cycle %d missing required fields "
                "(available: %s), skipping",
                i, list(data.keys()),
            )
            continue

        voltage = data[v_key]  # type: ignore[index]
        current = data[i_key]  # type: ignore[index]
        temperature = data[t_key]  # type: ignore[index]
        time_s = data[time_key]  # type: ignore[index]
        capacity = data[cap_key]  # type: ignore[index]

        # Capacity is the last value in the array (cumulative Ah)
        cap_value = float(capacity[-1]) if len(capacity) > 0 else np.nan

        # Handle missing/irregular timestamps
        if len(time_s) < 2:
            logger.warning("Cycle %d has < 2 time samples, skipping", i)
            continue

        # Check for non-monotonic timestamps and fix
        dt = np.diff(time_s)
        if np.any(dt <= 0):
            logger.warning(
                "Cycle %d has %d non-monotonic timestamps — "
                "forward-filling from last valid",
                i, int(np.sum(dt <= 0)),
            )
            # Replace non-monotonic points with linear interpolation
            good_mask = np.concatenate([[True], dt > 0])
            time_s = np.interp(
                np.arange(len(time_s)),
                np.where(good_mask)[0],
                time_s[good_mask],
            )

        # Ensure arrays are same length (truncate to shortest)
        min_len = min(len(voltage), len(current), len(temperature), len(time_s))
        if min_len < 2:
            logger.warning("Cycle %d has < 2 samples after alignment, skipping", i)
            continue

        discharge_cycles.append(CycleData(
            cycle_index=i,
            voltage=voltage[:min_len],
            current=current[:min_len],
            temperature=temperature[:min_len],
            time_seconds=time_s[:min_len],
            capacity_ah=cap_value,
            ambient_temp=cyc["ambient_temperature"],
            cycle_start_timestamp=cyc["time"],
        ))

    logger.info("Extracted %d discharge cycles", len(discharge_cycles))
    return discharge_cycles


def _find_key(data: dict[str, Any], candidates: list[str]) -> str | None:
    """Find the first matching key from candidates in a data dict."""
    for k in candidates:
        if k in data:
            return k
    return None


def compute_dq_dv(
    voltage: NDArray[np.float64],
    current: NDArray[np.float64],
    time_s: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    """Compute dQ/dV spectrum from a discharge cycle.

    Steps:
    1. Compute incremental charge: dQ = |I| * dt (Ah) via trapezoidal rule.
    2. Smooth Q-vs-V with Savitzky-Golay (window=SAVGOL_WINDOW_LENGTH,
       order=SAVGOL_POLY_ORDER).
    3. Numerically differentiate Q w.r.t. V.
    4. Interpolate onto uniform voltage bins (DQ_DV_VOLTAGE_BINS).

    Parameters
    ----------
    voltage : array
        Per-sample voltage (V), monotonically decreasing during discharge.
    current : array
        Per-sample current (A), negative during discharge.
    time_s : array
        Per-sample time (seconds).

    Returns
    -------
    dq_dv_spectrum : array of shape (N_DQ_DV_BINS,)
        dQ/dV at each voltage bin. NaN where voltage is outside discharge range.
    peak_voltage : float
        Voltage at dQ/dV maximum (V).
    peak_magnitude : float
        Value of dQ/dV at the peak (Ah/V).
    """
    # Incremental charge (Ah): integrate |I| over time
    dt = np.diff(time_s) / 3600.0  # seconds → hours
    q_incremental = np.abs(current[:-1]) * dt
    q_cumulative = np.concatenate([[0.0], np.cumsum(q_incremental)])

    # Sort by voltage (ascending) for interpolation — discharge V is decreasing
    sort_idx = np.argsort(voltage)
    v_sorted = voltage[sort_idx]
    q_sorted = q_cumulative[sort_idx]

    # Remove duplicate voltages
    unique_mask = np.concatenate([[True], np.diff(v_sorted) > 1e-6])
    v_sorted = v_sorted[unique_mask]
    q_sorted = q_sorted[unique_mask]

    if len(v_sorted) < SAVGOL_WINDOW_LENGTH:
        # Not enough points for Savitzky-Golay — return NaNs
        return (
            np.full(N_DQ_DV_BINS, np.nan),
            np.nan,
            np.nan,
        )

    # Smooth Q(V) with Savitzky-Golay before differentiation
    q_smooth = savgol_filter(
        q_sorted,
        window_length=min(SAVGOL_WINDOW_LENGTH, len(q_sorted) - (1 - len(q_sorted) % 2)),
        polyorder=SAVGOL_POLY_ORDER,
    )

    # Numerical derivative: dQ/dV
    dv = np.diff(v_sorted)
    dq = np.diff(q_smooth)
    # Avoid division by zero
    valid = np.abs(dv) > 1e-8
    dq_dv_raw = np.where(valid, dq / dv, 0.0)
    v_midpoints = 0.5 * (v_sorted[:-1] + v_sorted[1:])

    # Interpolate onto uniform voltage bins
    # Only interpolate within the actual voltage range
    v_min_actual = v_midpoints[0]
    v_max_actual = v_midpoints[-1]

    dq_dv_spectrum = np.full(N_DQ_DV_BINS, np.nan)
    in_range = (DQ_DV_VOLTAGE_BINS >= v_min_actual) & (
        DQ_DV_VOLTAGE_BINS <= v_max_actual
    )
    if np.any(in_range) and len(v_midpoints) >= 2:
        dq_dv_spectrum[in_range] = np.interp(
            DQ_DV_VOLTAGE_BINS[in_range], v_midpoints, dq_dv_raw
        )

    # Fill NaN at edges with zero (outside discharge voltage window)
    dq_dv_spectrum = np.nan_to_num(dq_dv_spectrum, nan=0.0)

    # Peak detection
    peak_idx = np.argmax(np.abs(dq_dv_spectrum))
    peak_voltage = float(DQ_DV_VOLTAGE_BINS[peak_idx])
    peak_magnitude = float(dq_dv_spectrum[peak_idx])

    return dq_dv_spectrum, peak_voltage, peak_magnitude


def fit_1rc_thevenin(
    voltage: NDArray[np.float64],
    current: NDArray[np.float64],
    time_s: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Fit 1RC Thevenin equivalent circuit model parameters per discharge cycle.

    Model: V(t) = V_ocv - I * R0 - I * R1 * (1 - exp(-t / (R1 * C1)))

    Estimation method:
    1. V_ocv: Voltage at the very start of discharge (first 3 samples averaged),
       which approximates the relaxed open-circuit voltage if the cell rested
       before this discharge. If no rest preceded, this is an approximation —
       but it's the best available without a separate OCV-SoC lookup table.
    2. R0: Instantaneous ohmic resistance from the initial voltage step when
       current is first applied: R0 = |V(t=0+) - V(t=0-)| / |I_step|.
    3. R1, C1: Nonlinear least-squares fit of the exponential transient
       V_transient(t) = V_ocv - I*R0 - I*R1*(1 - exp(-t/τ)) where τ = R1*C1.

    Fit requires ≥ ECM_FIT_MIN_SAMPLES samples. Returns NaN for R1, C1 if
    the fit fails or has insufficient data.

    Parameters
    ----------
    voltage, current, time_s : arrays
        Per-sample discharge data.

    Returns
    -------
    v_ocv, r0, r1, c1 : float
        ECM parameters. r1 and c1 may be NaN if fit fails.
    """
    if len(voltage) < ECM_FIT_MIN_SAMPLES:
        return np.nan, np.nan, np.nan, np.nan

    # V_ocv: average of first 3 samples (before significant IR drop develops)
    n_ocv = min(3, len(voltage))
    v_ocv = float(np.mean(voltage[:n_ocv]))

    # Mean current magnitude (discharge current is negative or positive
    # depending on convention — use absolute value)
    i_mean = float(np.mean(np.abs(current)))
    if i_mean < 1e-6:
        return v_ocv, np.nan, np.nan, np.nan

    # R0: ohmic drop — difference between V_ocv and voltage shortly after
    # current onset (sample index ~5 to avoid initial transient)
    r0_idx = min(5, len(voltage) - 1)
    v_after_ohmic = float(voltage[r0_idx])
    r0 = abs(v_ocv - v_after_ohmic) / i_mean
    # Clamp R0 to physically reasonable range [0.001, 1.0] Ω
    r0 = float(np.clip(r0, 0.001, 1.0))

    # Fit R1, C1 from the exponential transient
    # Use the region after the ohmic drop (samples r0_idx onward)
    t_fit = time_s[r0_idx:] - time_s[r0_idx]
    v_fit = voltage[r0_idx:]
    i_fit = np.abs(current[r0_idx:])

    if len(t_fit) < ECM_FIT_MIN_SAMPLES:
        return v_ocv, r0, np.nan, np.nan

    # Model: V(t) = V_ocv - I*R0 - I*R1*(1 - exp(-t/tau))
    # Rearrange: V_residual = V(t) - V_ocv + I*R0 = -I*R1*(1 - exp(-t/tau))
    v_residual = v_fit - v_ocv + i_fit * r0

    def _ecm_transient(t: NDArray, r1: float, tau: float) -> NDArray:
        return -np.mean(i_fit) * r1 * (1.0 - np.exp(-t / max(tau, 1e-6)))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _ecm_transient,
                t_fit,
                v_residual,
                p0=[0.05, 50.0],  # R1=50mΩ, τ=50s initial guess
                bounds=([0.001, 1.0], [2.0, 5000.0]),
                maxfev=2000,
            )
        r1 = float(popt[0])
        tau = float(popt[1])
        c1 = tau / max(r1, 1e-9)
        # Clamp to physically reasonable ranges
        r1 = float(np.clip(r1, 0.001, 2.0))
        c1 = float(np.clip(c1, 0.1, 100000.0))
    except (RuntimeError, ValueError) as e:
        logger.debug("ECM fit failed for cycle: %s", e)
        r1, c1 = np.nan, np.nan

    return v_ocv, r0, r1, c1


def compute_stressor_vector(
    current: NDArray[np.float64],
    temperature: NDArray[np.float64],
    time_s: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Compute the degradation stressor vector s_t for a single discharge cycle.

    This vector feeds ONLY the damage branch (§2.2). It is computed directly
    from raw electrical/thermal measurements and is deliberately NOT derived
    from the cross-attention backbone, preventing rest-related signal from
    leaking into the irreversible damage path (see §2.2 identifiability
    rationale).

    Parameters
    ----------
    current : array
        Per-sample current (A).
    temperature : array
        Per-sample temperature (°C).
    time_s : array
        Per-sample time (seconds).

    Returns
    -------
    throughput_cycle_ah : float
        Per-cycle incremental Ah-throughput: ∫|I|dt over THIS cycle only.
        NOT cumulative. Units: Ah.
    mean_temp_cycle : float
        Mean measured temperature during this discharge cycle (°C).
    c_rate : float
        Discharge C-rate: |I_mean| / RATED_CAPACITY. Dimensionless.
    """
    # Throughput: ∫|I|dt via trapezoidal integration, seconds → hours
    dt_hours = np.diff(time_s) / 3600.0
    throughput_cycle_ah = float(np.sum(np.abs(current[:-1]) * dt_hours))

    # Mean temperature
    mean_temp_cycle = float(np.mean(temperature))

    # C-rate
    c_rate = float(np.mean(np.abs(current))) / RATED_CAPACITY

    return throughput_cycle_ah, mean_temp_cycle, c_rate


def _compute_rest_metadata(
    all_cycles: list[dict[str, Any]],
    discharge_indices: list[int],
) -> list[tuple[float, float, float, int]]:
    """Compute rest-event metadata for each discharge cycle.

    Rest periods are the *gaps* between consecutive cycles in the raw data —
    they're not a separate cycle.type. We compute:
    - rest_duration: gap (hours) between this cycle's start and the end of
      the previous cycle
    - soc_at_rest_onset: SoC at end of previous cycle
    - ambient_temp_rest: average ambient temp of adjacent cycles
    - rest_flag: 1 if rest_duration > REST_THRESHOLD_HOURS

    Parameters
    ----------
    all_cycles : list[dict]
        ALL cycles (charge + discharge + impedance), not just discharges.
    discharge_indices : list[int]
        Indices into all_cycles corresponding to discharge cycles.

    Returns
    -------
    list[tuple]
        Per discharge cycle: (rest_duration_hours, soc_at_rest_onset,
        ambient_temp_rest, rest_flag).
    """
    results: list[tuple[float, float, float, int]] = []

    for pos, dis_idx in enumerate(discharge_indices):
        if dis_idx == 0 or pos == 0:
            # First cycle — no preceding rest
            results.append((0.0, np.nan, all_cycles[dis_idx]["ambient_temperature"], 0))
            continue

        # Find the immediately preceding cycle (of any type)
        prev_idx = dis_idx - 1
        prev_cycle = all_cycles[prev_idx]
        this_cycle = all_cycles[dis_idx]

        # Rest duration: gap between this cycle's start and previous cycle's end
        # Previous cycle end ≈ prev start + duration of prev cycle
        prev_data = prev_cycle["data"]
        prev_time_key = _find_key(prev_data, ["Time"])
        if prev_time_key is not None and len(prev_data[prev_time_key]) > 0:
            prev_duration_seconds = float(prev_data[prev_time_key][-1])
        else:
            prev_duration_seconds = 0.0

        # time field is now Unix timestamp (seconds) from _datevec_to_timestamp
        this_start_seconds = this_cycle["time"]
        prev_start_seconds = prev_cycle["time"]
        prev_end_seconds = prev_start_seconds + prev_duration_seconds

        rest_duration = max(0.0, (this_start_seconds - prev_end_seconds) / 3600.0)  # hours

        # SoC at rest onset: end-of-discharge SoC from previous discharge cycle
        # For simplicity, if prev cycle is discharge, SoC_eod ≈ 0 (full discharge)
        # If prev cycle is charge, SoC ≈ 1.0
        if prev_cycle["type"].lower() == "discharge":
            soc_at_rest = 0.0  # Full discharge → SoC ~ 0
        elif prev_cycle["type"].lower() == "charge":
            soc_at_rest = 1.0  # Full charge → SoC ~ 1
        else:
            soc_at_rest = np.nan  # Impedance or unknown

        # Ambient temp during rest: average of adjacent cycles
        amb_prev = prev_cycle["ambient_temperature"]
        amb_this = this_cycle["ambient_temperature"]
        ambient_temp_rest = float(np.mean([amb_prev, amb_this]))

        rest_flag = 1 if rest_duration > REST_THRESHOLD_HOURS else 0

        results.append((rest_duration, soc_at_rest, ambient_temp_rest, rest_flag))

    return results


def _compute_rul_labels(capacities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute RUL (Remaining Useful Life) labels for each discharge cycle.

    RUL_t = t_EOL - t, where t_EOL is the first cycle at which measured
    capacity drops below EOL_CAPACITY. If the cell never reaches EOL in the
    dataset (right-censored), RUL is set to NaN for all cycles — do NOT
    extrapolate or guess.

    Parameters
    ----------
    capacities : array
        Measured capacity (Ah) per discharge cycle.

    Returns
    -------
    rul : array
        RUL in cycles. NaN for right-censored cells.
    """
    # Find first cycle where capacity drops below EOL threshold
    eol_mask = capacities <= EOL_CAPACITY
    if not np.any(eol_mask):
        logger.warning(
            "Cell never reaches EOL (%.2f Ah). Min capacity: %.3f Ah. "
            "RUL labels will be NaN (right-censored).",
            EOL_CAPACITY, float(np.min(capacities)),
        )
        return np.full(len(capacities), np.nan)

    t_eol = int(np.argmax(eol_mask))  # first index where capacity <= EOL
    rul = np.array([t_eol - t for t in range(len(capacities))], dtype=np.float64)
    # Cycles after EOL: RUL = 0 (already past failure)
    rul[rul < 0] = 0.0
    return rul


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE: PER-CELL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cell_features(
    filepath: Path,
) -> pd.DataFrame:
    """Full feature extraction pipeline for one NASA PCoE battery cell.

    Loads the .mat file, extracts discharge cycles, computes all engineered
    features (dQ/dV, ECM, stressors, rest metadata), and returns a DataFrame
    indexed by discharge cycle number.

    Parameters
    ----------
    filepath : Path
        Path to the cell's .mat file.

    Returns
    -------
    pd.DataFrame
        One row per discharge cycle, columns matching CycleFeatures fields.
    """
    # Load raw cycles
    all_cycles = load_mat_file(filepath)

    # Extract discharge cycles
    discharge_cycles = _extract_discharge_cycles(all_cycles)
    if not discharge_cycles:
        logger.error("No discharge cycles found in %s", filepath.name)
        return pd.DataFrame()

    # Get discharge indices for rest metadata computation
    discharge_indices = [dc.cycle_index for dc in discharge_cycles]

    # Compute rest metadata
    rest_metadata = _compute_rest_metadata(all_cycles, discharge_indices)

    # Compute RUL labels
    capacities = np.array([dc.capacity_ah for dc in discharge_cycles])
    rul_labels = _compute_rul_labels(capacities)

    # Build feature rows
    rows: list[dict[str, Any]] = []

    for i, dc in enumerate(discharge_cycles):
        # ── Labels ──
        soh = dc.capacity_ah / RATED_CAPACITY
        # SoC at end of discharge: for full discharge, ~0; otherwise
        # approximate via Coulomb counting
        soc_eod = 0.0  # NASA PCoE cycles are full discharges to cutoff

        # ── Electrical summary stats ──
        v_mean = float(np.mean(dc.voltage))
        v_min = float(np.min(dc.voltage))
        v_max = float(np.max(dc.voltage))
        v_std = float(np.std(dc.voltage))
        # Voltage slope: linear fit V(t)
        if len(dc.time_seconds) > 1:
            coeffs = np.polyfit(dc.time_seconds, dc.voltage, deg=1)
            v_slope = float(coeffs[0])
        else:
            v_slope = 0.0

        i_mean = float(np.mean(dc.current))
        i_std = float(np.std(dc.current))

        # ── dQ/dV ──
        dq_dv_spectrum, dq_dv_peak_v, dq_dv_peak_mag = compute_dq_dv(
            dc.voltage, dc.current, dc.time_seconds
        )

        # ── ECM parameters ──
        v_ocv, r0, r1, c1 = fit_1rc_thevenin(
            dc.voltage, dc.current, dc.time_seconds
        )

        # ── Thermal stats ──
        t_mean = float(np.mean(dc.temperature))
        t_max = float(np.max(dc.temperature))
        t_std = float(np.std(dc.temperature))
        t_rise = float(dc.temperature[-1] - dc.temperature[0])

        # ── Stressor vector ──
        throughput, mean_temp, c_rate = compute_stressor_vector(
            dc.current, dc.temperature, dc.time_seconds
        )

        # ── Rest metadata ──
        rest_dur, soc_rest, amb_rest, r_flag = rest_metadata[i]

        row: dict[str, Any] = {
            "cycle_index": dc.cycle_index,
            "soh": soh,
            "soc_eod": soc_eod,
            "rul": rul_labels[i],
            "capacity_ah": dc.capacity_ah,
            # Electrical
            "v_mean": v_mean,
            "v_min": v_min,
            "v_max": v_max,
            "v_std": v_std,
            "v_slope": v_slope,
            "i_mean": i_mean,
            "i_std": i_std,
            # dQ/dV
            "dq_dv_peak_v": dq_dv_peak_v,
            "dq_dv_peak_mag": dq_dv_peak_mag,
            # ECM
            "v_ocv": v_ocv,
            "r0": r0,
            "r1": r1,
            "c1": c1,
            # Thermal
            "t_mean": t_mean,
            "t_max": t_max,
            "t_std": t_std,
            "t_rise": t_rise,
            # Stressor vector (damage branch input — NOT from backbone)
            "throughput_cycle_ah": throughput,
            "mean_temp_cycle": mean_temp,
            "c_rate": c_rate,
            # Rest metadata
            "rest_duration_hours": rest_dur,
            "soc_at_rest_onset": soc_rest,
            "ambient_temp_rest": amb_rest,
            "rest_flag": r_flag,
        }

        # dQ/dV spectrum as individual columns
        for j in range(N_DQ_DV_BINS):
            row[f"dq_dv_{j:02d}"] = dq_dv_spectrum[j]

        rows.append(row)

    df = pd.DataFrame(rows)
    df["cell_id"] = filepath.stem

    logger.info(
        "Cell %s: %d discharge cycles, SoH range [%.3f, %.3f], "
        "%d rest-flagged cycles (threshold=%.1f h)",
        filepath.stem,
        len(df),
        df["soh"].min(),
        df["soh"].max(),
        int(df["rest_flag"].sum()),
        REST_THRESHOLD_HOURS,
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# LOBO FOLD CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_lobo_folds(
    cell_ids: list[str] | None = None,
) -> list[tuple[list[str], str]]:
    """Build Leave-One-Battery-Out (LOBO) cross-validation folds.

    Each fold trains on N-1 cells and tests on the held-out cell. This is the
    only protocol that tests generalization given how few cells NASA PCoE has.

    Do NOT use chronological splits within one cell — that tests interpolation,
    not generalization.

    Parameters
    ----------
    cell_ids : list[str] or None
        Cell IDs to use. Defaults to CELL_IDS.

    Returns
    -------
    list[tuple[list[str], str]]
        Each tuple is (train_cell_ids, test_cell_id). Covers every cell
        exactly once as the test cell.
    """
    if cell_ids is None:
        cell_ids = CELL_IDS

    folds = []
    for test_cell in cell_ids:
        train_cells = [c for c in cell_ids if c != test_cell]
        folds.append((train_cells, test_cell))

    return folds


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING (ALL CELLS)
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_cells(
    data_dir: Path,
    cell_ids: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load and extract features for all NASA PCoE cells.

    Parameters
    ----------
    data_dir : Path
        Directory containing .mat files (e.g., data/raw/).
    cell_ids : list[str] or None
        Cell IDs to load. Defaults to CELL_IDS.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from cell_id to its feature DataFrame.
    """
    if cell_ids is None:
        cell_ids = CELL_IDS

    cell_data: dict[str, pd.DataFrame] = {}
    for cell_id in cell_ids:
        filepath = data_dir / f"{cell_id}.mat"
        if not filepath.exists():
            logger.error("File not found: %s — skipping cell %s", filepath, cell_id)
            continue
        df = extract_cell_features(filepath)
        if not df.empty:
            cell_data[cell_id] = df

    return cell_data


def get_lobo_splits(
    cell_data: dict[str, pd.DataFrame],
) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Convenience: combine load_all_cells + build_lobo_folds into ready splits.

    Parameters
    ----------
    cell_data : dict[str, pd.DataFrame]
        Output of load_all_cells().

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame, str]]
        Each tuple: (train_df, test_df, test_cell_id).
        train_df is the concatenation of all train cells' DataFrames.
    """
    folds = build_lobo_folds(list(cell_data.keys()))
    splits = []

    for train_ids, test_id in folds:
        if test_id not in cell_data:
            logger.warning("Test cell %s not in loaded data, skipping fold", test_id)
            continue
        train_dfs = [cell_data[cid] for cid in train_ids if cid in cell_data]
        if not train_dfs:
            logger.warning("No training data for fold with test=%s, skipping", test_id)
            continue
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = cell_data[test_id]
        splits.append((train_df, test_df, test_id))

    return splits


# ═══════════════════════════════════════════════════════════════════════════════
# CLI VERIFICATION (spec §9, step 1)
# ------------------------------------------------------------------------------

def _print_summary(cell_data: dict[str, pd.DataFrame]) -> None:
    """Print a summary table of loaded cell data for manual verification."""
    print("\n" + "=" * 80)
    print("PI-MDCNet v4 -- Data Pipeline Verification")
    print("=" * 80)

    for cell_id, df in cell_data.items():
        print(f"\n-- Cell: {cell_id} {'-' * 60}")
        print(f"  Discharge cycles:      {len(df)}")
        print(f"  SoH range:             [{df['soh'].min():.4f}, {df['soh'].max():.4f}]")
        print(f"  Capacity range (Ah):   [{df['capacity_ah'].min():.3f}, {df['capacity_ah'].max():.3f}]")

        rul_valid = df["rul"].dropna()
        if len(rul_valid) > 0:
            print(f"  RUL range (cycles):    [{rul_valid.min():.0f}, {rul_valid.max():.0f}]")
        else:
            print("  RUL:                   right-censored (never reached EOL)")

        n_rest = int(df["rest_flag"].sum())
        print(f"  Rest-flagged cycles:   {n_rest} / {len(df)} "
              f"(threshold={REST_THRESHOLD_HOURS:.1f} h)")

        if n_rest > 0:
            rest_df = df[df["rest_flag"] == 1]
            print(f"  Rest durations (h):    "
                  f"[{rest_df['rest_duration_hours'].min():.2f}, "
                  f"{rest_df['rest_duration_hours'].max():.2f}], "
                  f"mean={rest_df['rest_duration_hours'].mean():.2f}")

        print(f"  ECM R0 range (ohm):      [{df['r0'].min():.4f}, {df['r0'].max():.4f}]")
        r1_valid = df["r1"].dropna()
        if len(r1_valid) > 0:
            print(f"  ECM R1 range (ohm):      [{r1_valid.min():.4f}, {r1_valid.max():.4f}]")
        else:
            print("  ECM R1:                all fits failed")

        print(f"  Throughput range (Ah): "
              f"[{df['throughput_cycle_ah'].min():.3f}, "
              f"{df['throughput_cycle_ah'].max():.3f}]")
        print(f"  C-rate range:          [{df['c_rate'].min():.3f}, {df['c_rate'].max():.3f}]")
        print(f"  Temp range (C):        [{df['t_mean'].min():.1f}, {df['t_mean'].max():.1f}]")

    # LOBO folds
    folds = build_lobo_folds(list(cell_data.keys()))
    print(f"\n-- LOBO Folds ({len(folds)} folds) {'-' * 50}")
    for i, (train_ids, test_id) in enumerate(folds):
        print(f"  Fold {i}: train={train_ids}, test={test_id}")

    # Spot-check: verify rest_flag on a few cycles
    print(f"\n-- Rest-Flag Spot Check {'-' * 55}")
    for cell_id, df in cell_data.items():
        rest_cycles = df[df["rest_flag"] == 1].head(3)
        if len(rest_cycles) > 0:
            print(f"  {cell_id} -- first rest-flagged cycles:")
            for _, row in rest_cycles.iterrows():
                print(f"    cycle {int(row['cycle_index'])}: "
                      f"rest={row['rest_duration_hours']:.2f} h, "
                      f"soc_onset={row['soc_at_rest_onset']:.2f}, "
                      f"ambient={row['ambient_temp_rest']:.1f}C")
        else:
            print(f"  {cell_id} -- no rest-flagged cycles (all gaps < {REST_THRESHOLD_HOURS} h)")

    print("\n" + "=" * 80)
    print("Data pipeline verification complete.")
    print("=" * 80 + "\n")


def main() -> None:
    """CLI entry point for data pipeline verification (spec §9, step 1).

    Usage: python -m src.data_loader

    Loads all NASA PCoE cells, extracts features, prints summary, and
    spot-checks rest_flag against cycle metadata.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Default data directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"

    if not data_dir.exists():
        logger.error(
            "Data directory not found: %s\n"
            "Please download NASA PCoE battery .mat files to this location.\n"
            "Expected files: %s",
            data_dir,
            ", ".join(f"{cid}.mat" for cid in CELL_IDS),
        )
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        print("Download the NASA PCoE battery dataset and place .mat files in:")
        print(f"  {data_dir}")
        print(f"  Expected files: {', '.join(f'{cid}.mat' for cid in CELL_IDS)}")
        return

    cell_data = load_all_cells(data_dir)
    if not cell_data:
        logger.error("No cells loaded successfully.")
        return

    _print_summary(cell_data)

    # Quick assertion: verify column presence
    expected_cols = {
        "soh", "soc_eod", "rul", "v_mean", "v_min", "v_max", "v_std", "v_slope",
        "i_mean", "i_std", "dq_dv_peak_v", "dq_dv_peak_mag",
        "v_ocv", "r0", "r1", "c1",
        "t_mean", "t_max", "t_std", "t_rise",
        "throughput_cycle_ah", "mean_temp_cycle", "c_rate",
        "rest_duration_hours", "soc_at_rest_onset", "ambient_temp_rest", "rest_flag",
        "cell_id",
    }
    for cell_id, df in cell_data.items():
        missing = expected_cols - set(df.columns)
        assert not missing, f"Cell {cell_id} missing columns: {missing}"
    print("[PASS] All expected columns present in every cell DataFrame.")


if __name__ == "__main__":
    main()
