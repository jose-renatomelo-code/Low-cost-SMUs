from drivers import ADALM1000_Driver, AD3_Driver, Keithley2450_Driver, USMU_Driver
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INSTRUMENT SELECTION AND OUTPUT PATH  –  KEITHLEY / USMU / ADALM1000 / AD3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUMENT = "USMU"   # "USMU" | "KEITHLEY" | "ADALM1000" | "AD3"
OUTPUT_DIR  = Path("output JV")
OUTPUT_DIR.mkdir(exist_ok=True)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JV INITIAL PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCAN_RATES = [10, 1]      # V/s
N_LOOPS = 1
SWEEP_MODE = "rev/fwd"               # "rev/fwd" | "fwd/rev"
P_in = 100                           # mW/cm²
SAMPLE_AREA = 25                       # cm²
N_POINTS = 121
V_START = 0.0
V_STOP  = 1.1

# SS (steady-state) criteria params
VOC_WINDOW = 10
MIN_CV = 1e-3

# Modern plot style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f7f7fa",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e3e3e8",
    "grid.linestyle": "--",
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "font.size": 11,
    "lines.linewidth": 2.0,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DRIVER FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_driver(instrument: str):
    instrument = instrument.upper()
    if instrument == "USMU":
        return USMU_Driver("COM3", 9600, 0.01)
    if instrument == "KEITHLEY":
        return Keithley2450_Driver("USB0::0x05E6::0x2450::04387871::0::INSTR")
    if instrument == "ADALM1000":
        return ADALM1000_Driver()
    if instrument == "AD3":
        return AD3_Driver(scope_range_v=5.0, shunt_ohm=219.6)
    raise ValueError(
        f"Unknown instrument: {instrument!r}. Choose 'USMU', 'KEITHLEY', "
        f"'ADALM1000' or 'AD3'."
    )


def preconditioning_device(driver):
    """Check if Voc is stable by CV (<0.1%), save Voc transient fig and then JV is ready to start.

    The device must already be in CURRENT-SOURCE mode (set by main()). Here we
    only drive the sourcégés current to 0 and monitor the resulting Voc.
    """
    steady_state = False
    last_voc = 0
    last_voc_buffer, all_voc, ss_times = [], [], []
    t0 = time.perf_counter()
    # Voc measurement: source 0 A (current-source mode) and read the voltage.
    if hasattr(driver, "set_source_mode"):
        if INSTRUMENT == "KEITHLEY":
            driver.set_source_mode(curr_sour=True)
    # Re-baseline the current offset with the DUT connected (ADALM1000 only;
    # other drivers ignore it). Must happen before the sweep starts.
    if hasattr(driver, "calibrate_offset"):
        driver.calibrate_offset()
    print("Starting the preconditioning...")
    while not steady_state:
        try:
            if INSTRUMENT != "USMU":
                voc_read, c_read = driver.set_current_and_measure(0)
            else:
                voc_read, c_read = driver.set_voltage_and_measure(5)
            last_voc_buffer.append(voc_read)
            all_voc.append(voc_read)
            t_now = time.perf_counter() - t0
            ss_times.append(t_now)

            if len(last_voc_buffer) > VOC_WINDOW:
                del last_voc_buffer[0]
                voc_mean = np.mean(last_voc_buffer)
                voc_std = np.std(last_voc_buffer)
                if voc_mean != 0 and (voc_std / np.abs(voc_mean)) < MIN_CV:
                    total_ss_duration = time.perf_counter() - t0
                    print(f"Device reached steady-state in {total_ss_duration:.1f} seconds.")
                    last_voc = last_voc_buffer[-1]
                    last_voc_buffer.clear()
                    steady_state = True
        except Exception as e:
            print(f"SMU error on Voc read: {e}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ss_times, all_voc, color="gray", linestyle="--", linewidth=0.8, label="Voc Data")
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("Voc (V)", fontweight="bold")
    ax.set_title(f"Voc Evolution — {INSTRUMENT}", fontweight="bold")
    ax.legend(loc="best")
    smu_dir = OUTPUT_DIR / INSTRUMENT
    smu_dir.mkdir(parents=True, exist_ok=True)
    fig_path = smu_dir / "voc_transient.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Voc transient: {fig_path}")
    return last_voc


def calculate_pv_metrics(df):
    """PCE, Voc, Jsc, FF, Rs, Rsh."""
    if len(df) == 0:
        return {k: np.nan for k in
                ('Voc', 'Jsc', 'Vmpp', 'Jmpp', 'Pmax', 'FF', 'PCE', 'R_s', 'R_sh')}
    V = df["voltage(V)"].values
    J = df["j_current(mA/cm²)"].values
    P = df["power_density(mW/cm²)"].values

    # Voc - linear interpolation where J crosses 0
    sign_change = np.where(J[:-1] * J[1:] <= 0)[0]
    Voc = np.nan
    if len(sign_change) > 0:
        idx = sign_change[0]
        V_prox = V[idx:idx + 2]
        J_prox = J[idx:idx + 2]
        if len(J_prox) == 2:
            order = np.argsort(J_prox)
            Voc = np.interp(0.0, J_prox[order], V_prox[order])

    # Jsc - current at V=0
    def _interp_at_x(x_arr, y_arr, x0):
        if len(x_arr) < 2:
            return np.nan
        order = np.argsort(x_arr)
        xs = np.asarray(x_arr)[order]
        ys = np.asarray(y_arr)[order]
        if x0 < xs[0] or x0 > xs[-1]:
            return np.nan
        return float(np.interp(x0, xs, ys))

    Jsc_at_0 = _interp_at_x(V, J, 0.0)
    if np.isnan(Jsc_at_0) and len(V) > 0:
        Jsc_at_0 = J[np.argmin(np.abs(V))]
    Jsc = abs(float(Jsc_at_0)) if not np.isnan(Jsc_at_0) else np.nan

    # Pmax
    if np.all(np.isnan(P)):
        Pmax, Vmpp, Jmpp = np.nan, np.nan, np.nan
    else:
        idx_max = np.argmin(P)  # P typically negative for a generator
        Pmax = abs(float(P[idx_max]))
        Vmpp = float(V[idx_max])
        Jmpp = abs(float(J[idx_max]))

    # FF
    denom = (Voc * Jsc) if (not np.isnan(Voc) and not np.isnan(Jsc) and Voc != 0 and Jsc != 0) else np.nan
    FF = 100.0 * (Pmax / denom) if (denom and not np.isnan(Pmax)) else np.nan

    # PCE
    PCE = 100.0 * (Pmax / P_in) if (not np.isnan(Pmax) and P_in and P_in != 0) else np.nan

    # Resistances (Rs & Rsh)
    R_s = np.nan
    R_sh = np.nan
    if len(V) > 3 and not np.isnan(Voc) and not np.isnan(Jsc):
        try:
            I_amps = (J * SAMPLE_AREA) / 1000.0
            dI_dV = np.gradient(I_amps, V)
            idx_voc = (np.abs(V - Voc)).argmin()
            if dI_dV[idx_voc] != 0:
                R_s = abs(1.0 / dI_dV[idx_voc])
            idx_jsc = (np.abs(V - 0.0)).argmin()
            if dI_dV[idx_jsc] != 0:
                R_sh = abs(1.0 / dI_dV[idx_jsc])
        except Exception:
            pass

    return {'Voc': Voc, 'Jsc': Jsc, 'Vmpp': Vmpp, 'Jmpp': Jmpp, 'Pmax': Pmax,
            'FF': FF, 'PCE': PCE, 'R_s': R_s, 'R_sh': R_sh}


def plot_JV(df_full, smu_dir, instrument):
    """Modern JV plot with scan-rate colouring for each direction + power inset."""
    directions = ["fwd", "rev"]
    cmap = {"fwd": plt.cm.viridis, "rev": plt.cm.autumn}

    # Main J–V panel
    fig, ax = plt.subplots(figsize=(10, 7))
    for direction in directions:
        df_dir = df_full[df_full["direction"] == direction]
        if df_dir.empty:
            continue
        sr_unique = sorted(df_dir["scan_rate(V/s)"].unique())
        for i, sr in enumerate(sr_unique):
            sub = df_dir[df_dir["scan_rate(V/s)"] == sr].sort_values("voltage(V)")
            color = cmap[direction](i / max(len(sr_unique) - 1, 1))
            ax.plot(sub["voltage(V)"], sub["j_current(mA/cm²)"],
                    color=color, label=f"{direction.upper()} {sr:g} V/s")
    ax.set_xlabel("Voltage (V)", fontweight="bold")
    ax.set_ylabel("Current Density J (mA/cm²)", fontweight="bold")
    ax.set_title(f"J–V Curves — {instrument}", fontweight="bold")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.legend(loc="best", fontsize=9, ncol=2)

    # Power-density panel (per scan rate, rev direction highlighted)
    ax2 = ax.twinx()
    for direction in directions:
        df_dir = df_full[df_full["direction"] == direction]
        if df_dir.empty:
            continue
        sr_unique = sorted(df_dir["scan_rate(V/s)"].unique())
        for i, sr in enumerate(sr_unique):
            sub = df_dir[df_dir["scan_rate(V/s)"] == sr].sort_values("voltage(V)")
            color = cmap[direction](i / max(len(sr_unique) - 1, 1))
            ax2.plot(sub["voltage(V)"], sub["power_density(mW/cm²)"],
                     color=color, alpha=0.25, linestyle=":", linewidth=1.0)
    ax2.set_ylabel("Power Density (mW/cm²)", fontweight="bold", color="#555555")
    ax2.grid(False)

    fig.tight_layout()
    jv_path = smu_dir / "jv_curves.png"
    fig.savefig(jv_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved J–V plot: {jv_path}")

    # Separate P–V (power) plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for direction in directions:
        df_dir = df_full[df_full["direction"] == direction]
        if df_dir.empty:
            continue
        sr_unique = sorted(df_dir["scan_rate(V/s)"].unique())
        for i, sr in enumerate(sr_unique):
            sub = df_dir[df_dir["scan_rate(V/s)"] == sr].sort_values("voltage(V)")
            color = cmap[direction](i / max(len(sr_unique) - 1, 1))
            ax.plot(sub["voltage(V)"], sub["power_density(mW/cm²)"],
                    color=color, label=f"{direction.upper()} {sr:g} V/s")
    ax.set_xlabel("Voltage (V)", fontweight="bold")
    ax.set_ylabel("Power Density (mW/cm²)", fontweight="bold")
    ax.set_title(f"P–V Curve — {instrument}", fontweight="bold")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    pv_path = smu_dir / "pv_curve.png"
    fig.savefig(pv_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved P–V plot: {pv_path}")

    # Per scan-rate dedicated plots (each its own figure)
    for direction in directions:
        df_dir = df_full[df_full["direction"] == direction]
        if df_dir.empty:
            continue
        for sr in sorted(df_dir["scan_rate(V/s)"].unique()):
            sub = df_dir[df_dir["scan_rate(V/s)"] == sr].sort_values("voltage(V)")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(sub["voltage(V)"], sub["j_current(mA/cm²)"],
                    color="#1f77b4", label="J–V")
            ax.plot(sub["voltage(V)"], sub["power_density(mW/cm²)"],
                    color="#ff7f0e", label="P–V")
            m = calculate_pv_metrics(sub)
            ax.set_xlabel("Voltage (V)", fontweight="bold")
            ax.set_ylabel("J (mA/cm²)  /  P (mW/cm²)", fontweight="bold")
            ax.set_title(f"{instrument} — {direction.upper()} @ {sr:g} V/s\n"
                         f"PCE={m['PCE']:.2f}%  Voc={m['Voc']:.3f}V  "
                         f"Jsc={m['Jsc']:.2f}  FF={m['FF']:.1f}%",
                         fontweight="bold")
            ax.axhline(0, color="#999999", lw=0.8)
            ax.legend(loc="best")
            fig.tight_layout()
            sr_tag = f"{sr:g}".replace(".", "p")
            name = f"jv_{direction}_{sr_tag}Vps.png"
            fig.savefig(smu_dir / name, dpi=300, bbox_inches="tight")
            plt.close(fig)


def write_outputs(df_full, metrics_summary, smu_dir, instrument):
    """Write full CSV, metrics CSV, and a small per-scan-rate metrics breakdown."""
    full_path = smu_dir / "jv_full.csv"
    df_full.to_csv(full_path, index=False)
    print(f"Saved full CSV: {full_path}")

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_path = smu_dir / "jv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics CSV: {metrics_path}")

    # Per scan-rate / direction metrics (handy for HI analysis)
    rows = []
    for sr in sorted(df_full["scan_rate(V/s)"].unique()):
        for direction in ["fwd", "rev"]:
            sub = df_full[(df_full["scan_rate(V/s)"] == sr) & (df_full["direction"] == direction)]
            if sub.empty:
                continue
            m = calculate_pv_metrics(sub)
            rows.append({
                "scan_rate(V/s)": sr,
                "direction": direction,
                "PCE(%)": m["PCE"], "Voc(V)": m["Voc"],
                "Jsc(mA/cm²)": m["Jsc"], "FF(%)": m["FF"],
                "Pmax(mW/cm²)": m["Pmax"], "Rs(ohm)": m["R_s"],
                "Rsh(ohm)": m["R_sh"],
            })
    if rows:
        per_sr = pd.DataFrame(rows)
        per_sr_path = smu_dir / "jv_metrics_per_scanrate.csv"
        per_sr.to_csv(per_sr_path, index=False)
        print(f"Saved per-scan-rate metrics CSV: {per_sr_path}")


def main():
    print(f"\nConnecting to instrument: {INSTRUMENT}")
    driver = build_driver(INSTRUMENT)
    smu_dir = OUTPUT_DIR / INSTRUMENT / "6705"
    smu_dir.mkdir(parents=True, exist_ok=True)

    # 1) Connect in CURRENT-SOURCE mode (set Idrive=0, monitor Voc)
    driver.connect(curr_sour=True)

    # 2) Pre-conditioning: hold 0 A and watch Voc until it is stable.
    #    NB: the returned Voc MUST NOT redefine V_START/V_STOP (those are the
    #    dedicated sweep bounds, 0..1.1 V); we only run it for the settle wait.
    if SWEEP_MODE == "rev/fwd":
        try:
            preconditioning_device(driver)   # only used for the settle wait
        except Exception as e:
            print(f"Preconditioning skipped/failed: {e}")

    # 3) Switch to VOLTAGE-SOURCE mode for the JV sweep (no *RST, so settings
    #    configured below survive). This is the required reconfiguration step.
    if hasattr(driver, "set_source_mode"):
        driver.set_source_mode(curr_sour=False)
    if not INSTRUMENT == "ADALM1000":
        driver.connect(curr_sour=False)

    # 4) Integration / NPLC (applied AFTER mode switch so *RST cannot wipe it)
    if INSTRUMENT == "USMU":
        driver.configure_integration(1)
    elif INSTRUMENT == "KEITHLEY":
        driver.configure_integration(1)

    t0 = time.perf_counter()
    delta_V = np.abs(V_STOP - V_START)
    raw_data = []
    try:
        for loop in range(N_LOOPS):
            for sr in SCAN_RATES:
                # Total time to sweep ΔV at this scan rate, divided equally
                # across the N_POINTS voltage steps => dwell per point.
                t_dwell = delta_V / (sr * N_POINTS)
                if SWEEP_MODE in ("rev/fwd", "fwd/rev"):
                    seq = (
                        [(dv, "rev") for dv in np.linspace(V_STOP, V_START, N_POINTS)] +
                        [(dv, "fwd") for dv in np.linspace(V_START, V_STOP, N_POINTS)]
                    )
                else:
                    raise ValueError(f"Unsupported SWEEP_MODE: {SWEEP_MODE}")
                for dv, direction in seq:
                    # Apply the voltage for this step and measure one point.
                    v_meas, i_meas = driver.set_voltage_and_measure(dv)
                    j_meas = 1e3 * i_meas / SAMPLE_AREA          # mA/cm²
                    t_rel = float(time.perf_counter() - t0)
                    # Power density = V × J, with J in mA/cm² -> mW/cm²
                    p_mWcm2 = v_meas * j_meas
                    raw_data.append({
                        "time(s)": t_rel,
                        "loop": loop,
                        "direction": direction,
                        "voltage(V)": float(v_meas),
                        "j_current(mA/cm²)": float(j_meas),
                        "power_density(mW/cm²)": float(p_mWcm2),
                        "scan_rate(V/s)": sr,
                    })
                    print(f"Loop {loop} | {direction.upper()} | {sr:g} V/s | "
                          f"V = {v_meas:.3f} | J = {j_meas:.3f}")
                    # Hold this voltage until the dwell window elapses so the
                    # effective ramp rate matches the requested scan rate.
                    while (time.perf_counter() - t0) < (t_rel + t_dwell):
                        pass

    except Exception as e:
        print(f"Error while doing JV sweeping: {e}")
    finally:
        driver.disconnect()
        print("JV finished!")

    df_full = pd.DataFrame(raw_data)
    if df_full.empty:
        print("No data acquired; aborting output generation.")
        return

    plot_JV(df_full, smu_dir, INSTRUMENT)

    df_fwd = df_full[df_full["direction"] == "fwd"]
    df_rev = df_full[df_full["direction"] == "rev"]
    metrics_fwd = calculate_pv_metrics(df_fwd)
    metrics_rev = calculate_pv_metrics(df_rev)

    try:
        hi = 100 * (metrics_rev['PCE'] - metrics_fwd['PCE']) / (
            abs(metrics_fwd['PCE']) + 1e-12)
    except Exception:
        hi = np.nan

    metrics_summary = [{
        "Instrument": INSTRUMENT,
        "ScanRates(V/s)": ", ".join(f"{sr:g}" for sr in SCAN_RATES),
        "PCE_jv_fwd(%)": metrics_fwd['PCE'],
        "PCE_jv_rev(%)": metrics_rev['PCE'],
        "Voc_jv_fwd(V)": metrics_fwd['Voc'],
        "Voc_jv_rev(V)": metrics_rev['Voc'],
        "Jsc_fwd(mA/cm²)": metrics_fwd['Jsc'],
        "Jsc_rev(mA/cm²)": metrics_rev['Jsc'],
        "FF_fwd(%)": metrics_fwd['FF'],
        "FF_rev(%)": metrics_rev['FF'],
        "HI(%)": hi,
        "Rs(ohm)": metrics_rev['R_s'],
        "Rsh(ohm)": metrics_rev['R_sh'],
    }]

    write_outputs(df_full, metrics_summary, smu_dir, INSTRUMENT)


if __name__ == "__main__":
    main()