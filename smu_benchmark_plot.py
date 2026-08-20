"""
smu_benchmark_plot.py
=====================
Generates a single publication-quality figure comparing the measured current
step response across all characterised SMUs:
  - Keithley 2450  (NPLC = 0.01  – fastest setting)
  - uSMU Range 1   (OSR  = 1     – fastest setting)
  - ADALM1000      (SR   = 100 kSa/s)
  - AD3            (scope_rate = 100 kSa/s)

Each device's data are aligned so that t=0 corresponds to the rising-edge
voltage step.  Only the first positive half-period is shown (0 → HALF_PERIOD_S).

Output: output/graphs/smu_benchmark_current.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor":   "white",
    "axes.facecolor":     "#ffffff",
    "axes.edgecolor":     "#333333",
    "axes.linewidth":     1.2,
    "axes.grid":          True,
    "grid.color":         "#e9ecef",
    "grid.linestyle":     "--",
    "grid.alpha":         0.7,
    "axes.labelsize":     13,
    "axes.labelweight":   "bold",
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.frameon":     True,
    "legend.facecolor":   "white",
    "legend.edgecolor":   "#cccccc",
    "legend.fontsize":    11,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_DIR  = Path("output/csv")
OUT_DIR  = Path("output/graphs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── RC load used in the experiment (for theoretical curve) ────────────────────
R_OHM   = 9.90e3       # Ω
C_F     = 102.2e-6     # F
TAU_S   = R_OHM * C_F
V_STEP  = 2.0          # V

# ── Colour / marker palette (consistent with jv_comparison.py) ────────────────
SMU_STYLE = {
    "Keithley 2450\n(NPLC=0.01)": {
        "color": "#1f77b4", "lw": 2.0, "ls": "-",  "zorder": 5,
    },
    "uSMU Range 1\n(OSR=1)": {
        "color": "#2ca02c", "lw": 1.8, "ls": "-",  "zorder": 4,
    },
    "ADALM1000\n(SR=100 kSa/s)": {
        "color": "#ff7f0e", "lw": 1.8, "ls": "-",  "zorder": 3,
    },
    "AD3\n(SR=100 kSa/s)": {
        "color": "#9467bd", "lw": 1.8, "ls": "-",  "zorder": 2,
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_rise_edge(t: np.ndarray, v_set: np.ndarray, v_low=0.5) -> float:
    """
    Return the timestamp of the first rising edge where v_set crosses v_low
    from below.
    """
    above = v_set > v_low
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            # linear interpolation
            t0, t1 = t[i - 1], t[i]
            v0, v1 = v_set[i - 1], v_set[i]
            return t0 + (v_low - v0) / (v1 - v0) * (t1 - t0)
    # fallback
    return t[np.argmax(v_set > v_low)]


def load_and_align(csv_path: Path,
                   param_col: str,
                   param_val,
                   window_s: float = 5.0):
    """
    Load CSV, filter by param_col == param_val, find first rising edge,
    return (t_rel, i_mA) arrays trimmed to [0, window_s].
    """
    df = pd.read_csv(csv_path)

    # filter
    df = df[np.isclose(df[param_col], param_val, rtol=1e-3)].copy()
    df = df.dropna(subset=["time_s", "voltage_set_V", "current_A"])
    df = df.sort_values("time_s").reset_index(drop=True)

    t     = df["time_s"].to_numpy()
    v_set = df["voltage_set_V"].to_numpy()
    i_A   = df["current_A"].to_numpy()

    t_edge = _find_rise_edge(t, v_set)

    t_rel  = t - t_edge
    mask   = (t_rel >= -0.05) & (t_rel <= window_s)
    return t_rel[mask], i_A[mask] * 1e3   # → mA


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    WINDOW_S = 5.0   # seconds shown after the step

    datasets = [
        ("Keithley 2450\n(NPLC=0.01)",  CSV_DIR / "keithley_nplc_sweep.csv",  "nplc",       0.01),
        ("uSMU Range 1\n(OSR=1)",        CSV_DIR / "usmu_range1.csv",          "osr",        1),
        ("ADALM1000\n(SR=100 kSa/s)",   CSV_DIR / "adalm1000_sr_sweep.csv",   "sample_rate",100000),
        ("AD3\n(SR=100 kSa/s)",          CSV_DIR / "ad3_scope_rate_sweep.csv", "scope_rate", 100000),
    ]

    # Theoretical current step response
    t_th  = np.linspace(0, WINDOW_S, 2000)
    i_th  = (V_STEP / R_OHM) * np.exp(-t_th / TAU_S) * 1e3  # mA

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # theoretical
    ax.plot(t_th, i_th, "k--", lw=1.5, alpha=0.6, label="Theoretical\n(ideal RC)", zorder=6)

    for label, csv_path, param_col, param_val in datasets:
        try:
            t_rel, i_mA = load_and_align(csv_path, param_col, param_val, WINDOW_S)
        except Exception as e:
            print(f"  [WARN] Could not load {label}: {e}")
            continue

        style = SMU_STYLE[label]
        # Clean label for legend (replace \n with space)
        legend_label = label.replace("\n", " ")
        ax.plot(
            t_rel, i_mA,
            color=style["color"],
            lw=style["lw"],
            ls=style["ls"],
            label=legend_label,
            zorder=style["zorder"],
            alpha=0.92,
        )
        print(f"  Plotted: {legend_label}  ({len(t_rel)} pts)")

    # ── Decorations ───────────────────────────────────────────────────────────
    ax.axvline(0, color="#888888", lw=1.0, ls=":", alpha=0.8, label="Step edge")

    ax.set_xlabel("Relative time (s)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Current (mA)",      fontsize=13, fontweight="bold")
    ax.set_title(
        "SMU Benchmark – Measured Current Step Response\n"
        "(RC load: R ≈ 9.9 kΩ, C ≈ 102 µF, τ ≈ 1.01 s  |  0 V → 2 V step)",
        fontsize=13, fontweight="bold", pad=14, color="#111111"
    )
    ax.set_xlim(-0.1, WINDOW_S)
    ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="#cccccc",
    )
    ax.set_ylim(bottom=-0.02)

    plt.tight_layout()

    out_path = OUT_DIR / "smu_benchmark_current.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n  Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("=== SMU Benchmark – Current Step Response ===")
    main()
    print("=== Done ===")
