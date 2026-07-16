
import time
import csv
from pathlib import Path
from drivers import ADALM1000_Driver, AD3_Driver, Keithley2450_Driver, USMU_Driver
import numpy as np
import matplotlib.pyplot as plt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INSTRUMENT SELECTION  –  set INSTRUMENT to "USMU", "KEITHLEY" or "ADALM1000"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUMENT = "ADALM1000"           # "USMU" | "KEITHLEY" | "ADALM1000" | "AD3"

# uSMU connection
USMU_PORT         = "COM3"
USMU_BAUDRATE     = 9600
USMU_CMD_DELAY_S  = 0.01

# Keithley 2450 connection (VISA resource string, e.g. "USB0::0x05E6::0x2450::...")
KEITHLEY_VISA     = "USB0::0x05E6::0x2450::04387871::0::INSTR"

# ADALM1000
N_SETTLE_CALLS         = 3      # discard calls after constant()

SHUNT_RESISTANCE_OHM = 219.6

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STIMULUS / LOAD PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V_MAX_V         = 2.0         # square-wave high level  (V)
V_MIN_V         = 0.0        # square-wave low level   (V)
CURRENT_LIMIT_A = 0.1         # compliance current      (A)

# RC load nominal values (used for theoretical curve reference)
R_LOAD_OHM      = 9.9e3       # series resistance  (Ω)
C_LOAD_F        = 102.2e-6    # capacitance        (F)
TAU_S           = R_LOAD_OHM * C_LOAD_F   # time constant (s)

HALF_PERIOD_S   = 5 * TAU_S   # dwell time at each voltage level
N_HALF_PERIODS  = 6            # half-cycles per sweep run
RUN_DURATION_S  = N_HALF_PERIODS * HALF_PERIOD_S

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SWEEP PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# uSMU: outer loop = current range, inner loop = OSR
USMU_CURRENT_RANGES  = [1, 2, 3, 4]
USMU_OSR_VALUES      = [1, 10, 25, 50]

# Keithley: single sweep over NPLC values
KEITHLEY_NPLC_VALUES = [0.01, 0.1, 1, 10]

# ADALM1000: single sweep over session sample rate (Sa/s)
ADALM1000_SR  = [100000]

# AD3 (Analog Discovery 3): single sweep over scope sample rate (Sa/s)
AD3_SCOPE_RATE = [100000]

OUTPUT_DIR  = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_PATH = OUTPUT_DIR / f"report_{INSTRUMENT}.md"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ACQUISITION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def square_wave_voltage(t):
    t = np.asarray(t)
    half_cycle = np.floor(t / HALF_PERIOD_S).astype(int)
    return np.where(half_cycle % 2 == 0, V_MAX_V, V_MIN_V)


def acquire_segment(driver, sweep_param_value, t_offset_s: float) -> dict:
    """
    Acquire one time segment (RUN_DURATION_S) of square-wave data.

    Parameters
    ----------
    driver            : USMU_Driver or Keithley2450_Driver
    sweep_param_value : the parameter label (OSR or NPLC) already configured
    t_offset_s        : time offset to append to local timestamps

    Returns
    -------
    dict with keys: t, v_set, v_meas, i, sweep_param
    """
    t_list, v_set_list, v_meas_list, i_list, param_list = [], [], [], [], []

    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < RUN_DURATION_S:
        t_local = time.perf_counter() - t0
        v_set = square_wave_voltage(t_local)
        v_meas, i_meas = driver.set_voltage_and_measure(v_set)

        t_list.append(t_offset_s + t_local)
        v_set_list.append(v_set)
        v_meas_list.append(v_meas)
        i_list.append(i_meas)
        param_list.append(sweep_param_value)

    return {
        "t":           t_list,
        "v_set":       v_set_list,
        "v_meas":      v_meas_list,
        "i":           i_list,
        "sweep_param": param_list,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATISTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_threshold_crossing(t_arr, v_arr, threshold, direction="rising"):
    """
    Linear interpolation to find the exact time a signal crosses a threshold.
    direction : "rising" (signal crosses upward) | "falling" (downward)
    Returns None if threshold is never reached.
    """
    indices = (np.where(v_arr >= threshold)[0] if direction == "rising"
               else np.where(v_arr <= threshold)[0])
    if len(indices) == 0:
        return None
    idx = indices[0]
    if idx == 0:
        return t_arr[0]
    t0, t1 = t_arr[idx - 1], t_arr[idx]
    v0, v1 = v_arr[idx - 1], v_arr[idx]
    if abs(v1 - v0) < 1e-12:
        return t0
    return t0 + (t1 - t0) * (threshold - v0) / (v1 - v0)


def calculate_segment_statistics(t, v_set, v_meas, i) -> dict:
    """
    Compute characterization metrics for a single (instrument_param) segment.

    Metrics
    -------
    sample_rate_hz    : effective acquisition rate (Hz)
    std_v_overall_v   : overall voltage std dev (V)
    std_i_overall_a   : overall current std dev (A)
    std_v_noise_v     : steady-state voltage noise floor (V) – last 30% of each half-period
    std_i_noise_a     : steady-state current noise floor (A)
    rise_time_s       : 10%→90% voltage rise time from step onset (s)
    fall_time_s       : 90%→10% voltage fall time from step onset (s)
    """
    stats = {}
    n = len(t)

    # 1. Sample rate
    stats["sample_rate_hz"] = (1.0 / np.mean(np.diff(t))) if n > 1 else 0.0

    # 2. Overall std devs
    stats["std_v_overall_v"] = np.std(v_meas)
    stats["std_i_overall_a"] = np.std(i)

    # 3. Split into half-period intervals at v_set transitions
    trans_idx = np.where(v_set[:-1] != v_set[1:])[0]
    intervals = []
    s = 0
    for tx in trans_idx:
        intervals.append((s, tx))
        s = tx + 1
    intervals.append((s, n - 1))

    steady_v_res, steady_i_res = [], []
    rise_times, fall_times = [], []

    for j, (start, end) in enumerate(intervals):
        if end - start < 2:
            continue

        t_sub  = t[start:end + 1]
        v_sub  = v_meas[start:end + 1]
        i_sub  = i[start:end + 1]
        dur    = t_sub[-1] - t_sub[0]
        ss_mask = t_sub >= (t_sub[0] + 0.7 * dur)   # last 30% = steady-state

        # Steady-state noise floor
        if np.any(ss_mask):
            steady_v_res.extend(v_sub[ss_mask] - np.mean(v_sub[ss_mask]))
            steady_i_res.extend(i_sub[ss_mask] - np.mean(i_sub[ss_mask]))

        # Rise / fall times relative to step onset = prev_t[-1]
        if j > 0:
            ps, pe = intervals[j - 1]
            if pe - ps >= 2:
                prev_t   = t[ps:pe + 1]
                prev_v   = v_meas[ps:pe + 1]
                prev_dur = prev_t[-1] - prev_t[0]
                prev_ss  = prev_t >= (prev_t[0] + 0.7 * prev_dur)

                if np.any(prev_ss) and np.any(ss_mask):
                    v_base   = np.mean(prev_v[prev_ss])
                    v_target = np.mean(v_sub[ss_mask])
                    v_diff   = abs(v_target - v_base)

                    if v_diff > 0.5:
                        v_lo  = min(v_base, v_target)
                        v_hi  = max(v_base, v_target)
                        v10   = v_lo + 0.1 * v_diff
                        v90   = v_lo + 0.9 * v_diff
                        t_step = prev_t[-1]   # exact step onset

                        if v_target > v_base:   # rising edge
                            t90 = _find_threshold_crossing(t_sub, v_sub, v90, "rising")
                            if t90 is not None:
                                rise_times.append(t90 - t_step)
                        else:                   # falling edge
                            t10 = _find_threshold_crossing(t_sub, v_sub, v10, "falling")
                            if t10 is not None:
                                fall_times.append(t10 - t_step)

    stats["std_v_noise_v"] = np.std(steady_v_res) if steady_v_res else 0.0
    stats["std_i_noise_a"] = np.std(steady_i_res) if steady_i_res else 0.0
    stats["rise_time_s"]   = float(np.mean(rise_times)) if rise_times else float("nan")
    stats["fall_time_s"]   = float(np.mean(fall_times)) if fall_times else float("nan")

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOTTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Color palette – maps sweep param value to a plot color
_PARAM_COLORS = {
    # uSMU OSR
    1: "#d62728", 10: "#ff7f0e", 25: "#2ca02c", 50: "#1f77b4",
    # Keithley NPLC
    0.01: "#9467bd", 0.1: "#8c564b", 1: "#e377c2",
    # ADALM1000 sample rate (Sa/s)
    100000: "#17becf", 20000: "#bcbd22", 5000: "#ff9896", 1000: "#9467bd",
}
_DEFAULT_COLOR = "#7f7f7f"


def _theoretical_i_step(t_rel_arr: np.ndarray, v_before: float, v_after: float) -> np.ndarray:
    """
    Precise RC series-circuit current after a voltage step.

    For t < 0  : steady-state current = 0  (capacitor fully charged to v_before)
    For t >= 0 : I(t) = (v_after - v_before) / R · exp(−t / τ)

    This is exact for an ideal voltage source driving R in series with C.
    The capacitor starts at V_cap = v_before (steady state from the previous
    half-period) and charges toward v_after.
    """
    delta_v = v_after - v_before
    return np.where(
        t_rel_arr < 0,
        0.0,
        (delta_v / R_LOAD_OHM) * np.exp(-t_rel_arr / TAU_S)
    )


def plot_sweep_data(
    t, v_set, v_meas, i, sweep_param,
    outer_label: str,            # e.g. "Range 2"
    param_label: str,            # e.g. "OSR" or "NPLC"
    fig_path: Path,
):
    """
    2×2 panel figure:
      [0,0] Voltage full time-series
      [1,0] Current full time-series + theoretical RC curve
      [0,1] Voltage step-response zoom (first rising edge)
      [1,1] Current step-response zoom + theoretical RC curve
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_v, ax_i, ax_vz, ax_iz = axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1]

    unique_params = np.unique(sweep_param)

    # ── full time-series ──────────────────────────────────────────────────────
    ax_v.plot(t, v_set, color="gray", linestyle="--", linewidth=0.8, label="Setpoint")

    for p in unique_params:
        mask  = sweep_param == p
        color = _PARAM_COLORS.get(p, _DEFAULT_COLOR)
        ax_v.plot(t[mask], v_meas[mask], color=color, label=f"{param_label}={p}", alpha=0.85, linewidth=0.9)
        ax_i.plot(t[mask], i[mask] * 1e3, color=color, label=f"{param_label}={p}", alpha=0.85, linewidth=0.9)

    # Theoretical RC current: piecewise exponential, resets at each V_set transition.
    # Uses the exact previous steady-state voltage as the initial capacitor condition.
    # At t=0 (first half-period), the cap starts at 0 V.
    # At every subsequent transition: v_before = v_set of the previous half-period.
    i_theo = np.zeros_like(t)
    v_set_transitions = np.where(v_set[:-1] != v_set[1:])[0]
    seg_starts = np.concatenate(([0], v_set_transitions + 1))
    for k, s in enumerate(seg_starts):
        e = seg_starts[k + 1] if k + 1 < len(seg_starts) else len(t)
        t_local_seg = t[s:e] - t[s]
        v_after  = v_set[s]
        v_before = 0.0 if k == 0 else v_set[seg_starts[k - 1]]
        i_theo[s:e] = (v_after - v_before) / R_LOAD_OHM * np.exp(-t_local_seg / TAU_S)
    ax_i.plot(t, i_theo * 1e3, color="black", linestyle="--", linewidth=1.1, label="Theoretical (tr=0)")

    ax_v.set_ylabel("Voltage (V)", fontweight="bold")
    ax_v.set_title(f"Voltage – {outer_label}", fontweight="bold")
    ax_v.grid(True, linestyle="--", alpha=0.4)
    ax_v.legend(fontsize=8)

    ax_i.set_ylabel("Current (mA)", fontweight="bold")
    ax_i.set_xlabel("Time (s)", fontweight="bold")
    ax_i.set_title(f"Current – {outer_label}", fontweight="bold")
    ax_i.grid(True, linestyle="--", alpha=0.4)
    ax_i.legend(fontsize=8)

    # ── step-response zoom ────────────────────────────────────────────────────
    # Compute per-parameter median dt, then size the zoom window so the
    # SLOWEST parameter shows at least 15 points after the step.
    param_dt = {}
    for p in unique_params:
        t_p = t[sweep_param == p]
        param_dt[p] = float(np.median(np.diff(t_p))) if len(t_p) > 1 else 1.0

    max_dt      = max(param_dt.values())          # dt of the slowest param
    zoom_pre_s  = max(0.05 * TAU_S, 5 * max_dt)  # show a few pre-step points
    zoom_post_s = max(3.0 * TAU_S, 15 * max_dt)  # show ≥15 post-step points

    for p in unique_params:
        mask   = sweep_param == p
        t_seg  = t[mask]
        vs_seg = v_set[mask]
        vm_seg = v_meas[mask]
        i_seg  = i[mask]
        color  = _PARAM_COLORS.get(p, _DEFAULT_COLOR)

        # Detect genuine rising transitions *within* this parameter's data.
        # Inter-segment boundaries have dt >> local median → excluded.
        dt_seg   = np.diff(t_seg)
        med_dt_p = param_dt[p]
        raw_rising = np.where((vs_seg[:-1] < vs_seg[1:]) &
                              (dt_seg <= 2.0 * med_dt_p))[0]
        if len(raw_rising) == 0:
            continue

        t_step = t_seg[raw_rising[0]]
        t_rel  = t_seg - t_step
        zm     = (t_rel >= -zoom_pre_s) & (t_rel <= zoom_post_s)

        # Scale marker so sparse (slow) curves remain visible
        msize = max(3.0, min(8.0, 2.0 / med_dt_p ** 0.3))

        if np.any(zm):
            ax_vz.plot(t_rel[zm], vm_seg[zm], color=color, label=f"{param_label}={p}",
                       linewidth=1.4, marker="o", markersize=msize)
            ax_iz.plot(t_rel[zm], i_seg[zm] * 1e3, color=color, label=f"{param_label}={p}",
                       linewidth=1.4, marker="o", markersize=msize)
            if p == unique_params[0]:
                ax_vz.plot(t_rel[zm], square_wave_voltage(t_rel[zm]),
                           color="black", linestyle="--", linewidth=1.2, label="Theoretical V (tr=0)")

    # Theoretical zoom curves (precise RC model, rising from V_MIN to V_MAX)
    t_rel_theo = np.linspace(-zoom_pre_s, zoom_post_s, 1000)
    ax_iz.plot(t_rel_theo, _theoretical_i_step(t_rel_theo, V_MIN_V, V_MAX_V) * 1e3,
               color="black", linestyle="--", linewidth=1.2, label="Theoretical I (tr=0)")
    # Enforce axis limits — prevents outlier samples from stretching the scale
    v_margin = 0.15 * (V_MAX_V - V_MIN_V)
    ax_vz.set_xlim(-0.1, 5.1)
    ax_vz.set_ylim(V_MIN_V - v_margin, V_MAX_V + v_margin)
    ax_iz.set_xlim(-0.1, 5.1)

    ax_vz.axvline(0, color="gray", linestyle=":", linewidth=0.9, label="Step edge")
    ax_iz.axvline(0, color="gray", linestyle=":", linewidth=0.9)

    ax_vz.set_ylabel("Voltage (V)", fontweight="bold")
    ax_vz.set_xlabel("Relative time (s)", fontweight="bold")
    ax_vz.set_title("Step Response – Voltage", fontweight="bold")
    ax_vz.grid(True, linestyle="--", alpha=0.4)
    ax_vz.legend(fontsize=8, loc="lower right")

    ax_iz.set_ylabel("Current (mA)", fontweight="bold")
    ax_iz.set_xlabel("Relative time (s)", fontweight="bold")
    ax_iz.set_title("Step Response – Current", fontweight="bold")
    ax_iz.grid(True, linestyle="--", alpha=0.4)
    ax_iz.legend(fontsize=8, loc="upper right")

    if INSTRUMENT == "USMU":
        instrument_name = "usmu"
    elif INSTRUMENT == "KEITHLEY":
        instrument_name = "keithley"
    elif INSTRUMENT == "ADALM1000":
        instrument_name = "adalm1000"
    elif INSTRUMENT == "AD3":
        instrument_name = "ad3"
    fig.suptitle(f"{instrument_name} Characterization  –  {outer_label}",
                 fontsize=14, fontweight="bold")
    fig.savefig(fig_path, dpi=300)

    try:
        plt.show()
    except Exception as exc:
        print(f"  (plt.show() unavailable: {exc})")

    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_csv(path: Path, t, v_set, v_meas, i, sweep_param, param_label: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "voltage_set_V", "voltage_meas_V", "current_A", param_label])
        for row in zip(t, v_set, v_meas, i, sweep_param):
            writer.writerow(row)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_report(all_stats: list, instrument: str, report_path: Path):
    if instrument == "USMU":
        instrument_name = "uSMU"
    elif instrument == "ADALM1000":
        instrument_name = "ADALM1000"
    elif instrument == "AD3":
        instrument_name = "Analog Discovery 3"
    else:
        instrument_name = "Keithley 2450"

    lines = [
        f"# {instrument_name} Characterization Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Measurement Conditions",
        f"- **Stimulus**: Square wave  {V_MIN_V} V ↔ {V_MAX_V} V",
        f"- **Half-period**: {HALF_PERIOD_S:.3f} s  (= {HALF_PERIOD_S/TAU_S:.0f} × τ)",
        f"- **Load (nominal)**: R = {R_LOAD_OHM/1e3:.2f} kΩ,  C = {C_LOAD_F*1e6:.1f} µF,  τ = {TAU_S:.3f} s",
        f"- **Compliance current**: {CURRENT_LIMIT_A*1e3:.0f} mA\n",
        "## Statistics",
    ]

    if instrument == "USMU":
        lines.append(
            "| Range | OSR | Sample Rate (Hz) | V noise (mV rms) | "
            "I noise (µA rms) | Rise time (ms) | Fall time (ms) |"
        )
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for s in all_stats:
            tr = f"{s['rise_time_s']*1e3:.2f}" if not np.isnan(s["rise_time_s"]) else "N/A"
            tf = f"{s['fall_time_s']*1e3:.2f}" if not np.isnan(s["fall_time_s"]) else "N/A"
            lines.append(
                f"| {s['range']} | {s['osr']} "
                f"| {s['sample_rate_hz']:.2f} "
                f"| {s['std_v_noise_v']*1e3:.3f} "
                f"| {s['std_i_noise_a']*1e6:.3f} "
                f"| {tr} | {tf} |"
            )
    elif instrument in ("ADALM1000", "AD3"):
        rate_label = "Sample Rate (Sa/s)" if instrument == "ADALM1000" else "Scope Rate (Sa/s)"
        lines.append(
            f"| {rate_label} | Sample Rate (Hz) | V noise (mV rms) | "
            "I noise (µA rms) | Rise time (ms) | Fall time (ms) |"
        )
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
        for s in all_stats:
            tr = f"{s['rise_time_s']*1e3:.2f}" if not np.isnan(s["rise_time_s"]) else "N/A"
            tf = f"{s['fall_time_s']*1e3:.2f}" if not np.isnan(s["fall_time_s"]) else "N/A"
            lines.append(
                f"| {s['sample_rate']} "
                f"| {s['sample_rate_hz']:.2f} "
                f"| {s['std_v_noise_v']*1e3:.3f} "
                f"| {s['std_i_noise_a']*1e6:.3f} "
                f"| {tr} | {tf} |"
            )
    else:  # Keithley 2450
        lines.append(
            "| NPLC | Sample Rate (Hz) | V noise (mV rms) | "
            "I noise (µA rms) | Rise time (ms) | Fall time (ms) |"
        )
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
        for s in all_stats:
            tr = f"{s['rise_time_s']*1e3:.2f}" if not np.isnan(s["rise_time_s"]) else "N/A"
            tf = f"{s['fall_time_s']*1e3:.2f}" if not np.isnan(s["fall_time_s"]) else "N/A"
            lines.append(
                f"| {s['nplc']} "
                f"| {s['sample_rate_hz']:.2f} "
                f"| {s['std_v_noise_v']*1e3:.3f} "
                f"| {s['std_i_noise_a']*1e6:.3f} "
                f"| {tr} | {tf} |"
            )

    lines += [
        "\n## Notes",
        "> [!NOTE]",
        "> **Noise floor** is the std dev of residuals in the last 30 % of each half-period (settled region).",
        "",
        "> [!NOTE]",
        "> **Rise / fall times** are measured from the exact voltage-step onset to the 90 % (rise) or 10 % (fall) "
        "threshold crossing, using linear interpolation between samples.",
    ]

    content = "\n".join(lines)
    report_path.write_text(content, encoding="utf-8")
    print(f"\nReport saved to: {report_path.resolve()}")
    print("\n" + content)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ── connect instrument ────────────────────────────────────────────────────
    print(f"\nConnecting to instrument: {INSTRUMENT}")

    if INSTRUMENT == "USMU":
        driver = USMU_Driver(USMU_PORT, USMU_BAUDRATE, USMU_CMD_DELAY_S)
    elif INSTRUMENT == "KEITHLEY":
        driver = Keithley2450_Driver(KEITHLEY_VISA)
    elif INSTRUMENT == "ADALM1000":
        driver = ADALM1000_Driver()
    elif INSTRUMENT == "AD3":
        driver = AD3_Driver(
            scope_range_v=5.0,            # 5 V full-scale per analog-input channel
            shunt_ohm=SHUNT_RESISTANCE_OHM,
        )
    else:
        raise ValueError(
            f"Unknown instrument: {INSTRUMENT!r}. Choose 'USMU', 'KEITHLEY', "
            f"'ADALM1000' or 'AD3'."
        )

    driver.connect()

    all_stats = []

    try:
        # ── uSMU: outer loop = current range ──────────────────────────────────
        if INSTRUMENT == "USMU":
            for current_range in USMU_CURRENT_RANGES:
                print(f"\n── Range {current_range} {'─'*50}")
                driver.configure_range(current_range)

                # Accumulate all OSR traces into one combined dataset per range
                seg_t, seg_vs, seg_vm, seg_i, seg_param = [], [], [], [], []
                t_offset = 0.0

                for osr in USMU_OSR_VALUES:
                    print(f"   OSR={osr:3d}  ({RUN_DURATION_S:.1f} s) ...", end=" ", flush=True)
                    driver.configure_integration(osr)

                    seg = acquire_segment(driver, osr, t_offset)

                    stats = calculate_segment_statistics(
                        np.array(seg["t"]), np.array(seg["v_set"]),
                        np.array(seg["v_meas"]), np.array(seg["i"]),
                    )
                    stats.update(range=current_range, osr=osr)
                    all_stats.append(stats)
                    print(f"fs={stats['sample_rate_hz']:.1f} Hz  "
                          f"V_noise={stats['std_v_noise_v']*1e3:.2f} mV  "
                          f"I_noise={stats['std_i_noise_a']*1e6:.1f} µA")

                    seg_t.extend(seg["t"])
                    seg_vs.extend(seg["v_set"])
                    seg_vm.extend(seg["v_meas"])
                    seg_i.extend(seg["i"])
                    seg_param.extend(seg["sweep_param"])
                    t_offset = seg["t"][-1] if seg["t"] else t_offset

                prefix    = f"usmu_range{current_range}"
                csv_path  = OUTPUT_DIR / f"{prefix}.csv"
                fig_path  = OUTPUT_DIR / f"{prefix}.png"

                save_csv(csv_path, np.array(seg_t), np.array(seg_vs),
                         np.array(seg_vm), np.array(seg_i),
                         np.array(seg_param), "osr")
                print(f"  → CSV : {csv_path}")

                plot_sweep_data(
                    np.array(seg_t), np.array(seg_vs),
                    np.array(seg_vm), np.array(seg_i),
                    np.array(seg_param),
                    outer_label=f"Range {current_range}",
                    param_label="OSR",
                    fig_path=fig_path,
                )
                print(f"  → Plot: {fig_path}")

        # ── Keithley: single sweep over NPLC ─────────────────────────────────
        elif INSTRUMENT == "KEITHLEY":
            print(f"\n── Keithley 2450  NPLC sweep {'─'*40}")

            seg_t, seg_vs, seg_vm, seg_i, seg_param = [], [], [], [], []
            t_offset = 0.0

            for nplc in KEITHLEY_NPLC_VALUES:
                print(f"   NPLC={nplc}  ({RUN_DURATION_S:.1f} s) ...", end=" ", flush=True)
                driver.configure_integration(nplc)

                seg = acquire_segment(driver, nplc, t_offset)

                stats = calculate_segment_statistics(
                    np.array(seg["t"]), np.array(seg["v_set"]),
                    np.array(seg["v_meas"]), np.array(seg["i"]),
                )
                stats.update(nplc=nplc)
                all_stats.append(stats)
                print(f"fs={stats['sample_rate_hz']:.1f} Hz  "
                      f"V_noise={stats['std_v_noise_v']*1e3:.2f} mV  "
                      f"I_noise={stats['std_i_noise_a']*1e6:.1f} µA")

                seg_t.extend(seg["t"])
                seg_vs.extend(seg["v_set"])
                seg_vm.extend(seg["v_meas"])
                seg_i.extend(seg["i"])
                seg_param.extend(seg["sweep_param"])
                t_offset = seg["t"][-1] if seg["t"] else t_offset

            csv_path = OUTPUT_DIR / "keithley_nplc_sweep.csv"
            fig_path = OUTPUT_DIR / "keithley_nplc_sweep.png"

            save_csv(csv_path, np.array(seg_t), np.array(seg_vs),
                     np.array(seg_vm), np.array(seg_i),
                     np.array(seg_param), "nplc")
            print(f"  → CSV : {csv_path}")

            plot_sweep_data(
                np.array(seg_t), np.array(seg_vs),
                np.array(seg_vm), np.array(seg_i),
                np.array(seg_param),
                outer_label="NPLC Sweep",
                param_label="NPLC",
                fig_path=fig_path,
            )
            print(f"  → Plot: {fig_path}")

        # ── ADALM1000: single sweep over sample rate ───────────────────────
        elif INSTRUMENT == "ADALM1000":
            print(f"\n── ADALM1000  sample-rate sweep {'─'*36}")

            seg_t, seg_vs, seg_vm, seg_i, seg_param = [], [], [], [], []
            t_offset = 0.0

            for sr in ADALM1000_SR:
                print(f"   SR={sr} Sa/s  ({RUN_DURATION_S:.1f} s) ...", end=" ", flush=True)
                driver.configure_integration(sr)

                seg = acquire_segment(driver, sr, t_offset)

                stats = calculate_segment_statistics(
                    np.array(seg["t"]), np.array(seg["v_set"]),
                    np.array(seg["v_meas"]), np.array(seg["i"]),
                )
                stats.update(sample_rate=sr)
                all_stats.append(stats)
                print(f"fs={stats['sample_rate_hz']:.1f} Hz  "
                      f"V_noise={stats['std_v_noise_v']*1e3:.2f} mV  "
                      f"I_noise={stats['std_i_noise_a']*1e6:.1f} µA")

                seg_t.extend(seg["t"])
                seg_vs.extend(seg["v_set"])
                seg_vm.extend(seg["v_meas"])
                seg_i.extend(seg["i"])
                seg_param.extend(seg["sweep_param"])
                t_offset = seg["t"][-1] if seg["t"] else t_offset

            csv_path = OUTPUT_DIR / "adalm1000_sr_sweep.csv"
            fig_path = OUTPUT_DIR / "adalm1000_sr_sweep.png"

            save_csv(csv_path, np.array(seg_t), np.array(seg_vs),
                     np.array(seg_vm), np.array(seg_i),
                     np.array(seg_param), "sample_rate")
            print(f"  → CSV : {csv_path}")

            plot_sweep_data(
                np.array(seg_t), np.array(seg_vs),
                np.array(seg_vm), np.array(seg_i),
                np.array(seg_param),
                outer_label="Sample Rate Sweep",
                param_label="SR (Sa/s)",
                fig_path=fig_path,
            )
            print(f"  → Plot: {fig_path}")

        # ── AD3: single sweep over scope sample rate ────────────────────────
        elif INSTRUMENT == "AD3":
            print(f"\n── AD3  scope-rate sweep {'─'*40}")

            seg_t, seg_vs, seg_vm, seg_i, seg_param = [], [], [], [], []
            t_offset = 0.0

            for sr in AD3_SCOPE_RATE:
                print(f"   scope_rate={sr} Sa/s  ({RUN_DURATION_S:.1f} s) ...",
                      end=" ", flush=True)
                driver.configure_integration(sr)

                seg = acquire_segment(driver, sr, t_offset)

                stats = calculate_segment_statistics(
                    np.array(seg["t"]), np.array(seg["v_set"]),
                    np.array(seg["v_meas"]), np.array(seg["i"]),
                )
                stats.update(sample_rate=sr)
                all_stats.append(stats)
                print(f"fs={stats['sample_rate_hz']:.1f} Hz  "
                      f"V_noise={stats['std_v_noise_v']*1e3:.2f} mV  "
                      f"I_noise={stats['std_i_noise_a']*1e6:.1f} µA")

                seg_t.extend(seg["t"])
                seg_vs.extend(seg["v_set"])
                seg_vm.extend(seg["v_meas"])
                seg_i.extend(seg["i"])
                seg_param.extend(seg["sweep_param"])
                t_offset = seg["t"][-1] if seg["t"] else t_offset

            csv_path = OUTPUT_DIR / "ad3_scope_rate_sweep.csv"
            fig_path = OUTPUT_DIR / "ad3_scope_rate_sweep.png"

            save_csv(csv_path, np.array(seg_t), np.array(seg_vs),
                     np.array(seg_vm), np.array(seg_i),
                     np.array(seg_param), "scope_rate")
            print(f"  → CSV : {csv_path}")

            plot_sweep_data(
                np.array(seg_t), np.array(seg_vs),
                np.array(seg_vm), np.array(seg_i),
                np.array(seg_param),
                outer_label="Scope Rate Sweep",
                param_label="SR (Sa/s)",
                fig_path=fig_path,
            )
            print(f"  → Plot: {fig_path}")

        # ── report ────────────────────────────────────────────────────────────
        generate_report(all_stats, INSTRUMENT, REPORT_PATH)

    finally:
        if driver is not None:
            driver.disconnect()
        print("\nInstrument disconnected.")


if __name__ == "__main__":
    main()