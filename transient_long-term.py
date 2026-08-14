import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

from drivers import MultiADALM1000_Driver, ADALM1000_Driver
from mppt import double_exp_fitting, calculate_r_squared

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION & PARAMETERS (Apenas Canal A por ADALM1000)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V_START               = 0.6          # Initial tracking voltage (V)
LARGE_STEP            = 0.1          # Exploration step size (V) for transient tracking
SMALL_STEP            = 0.05         # Refinement step size (V)
SAMPLE_AREA           = 0.16         # Active area (cm²)
P_IN                  = 100.0        # Incident light intensity (mW/cm²)
T_TOTAL               = 7 * 24 * 3600  # Total duration (seconds): 1 week
HOURLY_CHECK_INTERVAL = 3600         # Interval for transient analysis checkpoint (seconds: 1h)
N_TRANSIENT_CYCLES    = 10           # Number of transient cycles per hourly evaluation
TIMEOUT_CV            = 15.0         # Max wait time for CV stabilization per cycle (seconds)
CV_WINDOW             = 20           # Sliding window size for CV check
MIN_CV                = 0.001        # Max CV threshold (0.1%) for steady state

BASE_OUTPUT_DIR = Path("OUTPUT STABILITY")
BASE_OUTPUT_DIR.mkdir(exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ESTILO DE PLOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f7f7fa",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e3e3e8",
    "grid.linestyle": "--",
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "font.size": 10,
    "lines.linewidth": 1.8,
    "legend.frameon": True,
    "legend.facecolor": "white",
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def po_step(p_now: float, v_now: float, direction: int, is_exploring: bool, p_prev: float) -> tuple[float, int, bool]:
    """Perturb & Observe MPPT logic with large/small step adaptation."""
    if p_prev != 0.0:
        dP = p_now - p_prev
        if dP < 0:
            direction *= -1
            if is_exploring:
                is_exploring = False

    step = LARGE_STEP if is_exploring else SMALL_STEP
    v_next = v_now + direction * step
    v_next = float(np.clip(v_next, 0.0, 1.3))  # Safe ADALM1000 voltage bounds
    return v_next, direction, is_exploring


def acquire_cv_transient_cha(
    driver: MultiADALM1000_Driver,
    v_targets: dict,
    timeout: float = TIMEOUT_CV
) -> tuple[dict, dict]:
    """
    Applies v_targets (dev_id -> v_target) on Channel A for all devices and acquires samples
    until steady state is reached (CV < MIN_CV) or timeout is reached.

    Returns:
    - result_window: dict mapping dev_id -> (t_arr, v_arr, i_arr)
    - steady_stats: dict mapping dev_id -> (v_ss_avg, i_ss_avg)
    """
    start_t = time.perf_counter()
    dev_keys = list(driver.devices.keys())

    data = {dev_id: {"t": [], "v": [], "i": []} for dev_id in dev_keys}
    cv_buffers = {dev_id: [] for dev_id in dev_keys}
    settled = {dev_id: False for dev_id in dev_keys}

    while (time.perf_counter() - start_t) < timeout and not all(settled.values()):
        meas = driver.set_voltage_channel_a_all_and_measure(v_targets)
        t_rel = time.perf_counter() - start_t

        for dev_id, (v_a, i_a) in meas.items():
            data[dev_id]["t"].append(t_rel)
            data[dev_id]["v"].append(v_a)
            data[dev_id]["i"].append(i_a)
            cv_buffers[dev_id].append(i_a)

            if len(cv_buffers[dev_id]) > CV_WINDOW:
                cv_buffers[dev_id].pop(0)
                m = np.mean(cv_buffers[dev_id])
                s = np.std(cv_buffers[dev_id])
                if m != 0 and (s / abs(m)) < MIN_CV:
                    settled[dev_id] = True

    result_window = {}
    steady_stats = {}
    for dev_id, val in data.items():
        t_arr = np.array(val["t"])
        v_arr = np.array(val["v"])
        i_arr = np.array(val["i"])
        result_window[dev_id] = (t_arr, v_arr, i_arr)

        # Calculate steady-state average from last 30% of settled window
        n_pts = len(v_arr)
        ss_start = int(0.7 * n_pts) if n_pts >= 5 else 0
        v_ss = float(np.mean(v_arr[ss_start:])) if n_pts > 0 else 0.0
        i_ss = float(np.mean(i_arr[ss_start:])) if n_pts > 0 else 0.0
        steady_stats[dev_id] = (v_ss, i_ss)

    return result_window, steady_stats


def print_mppt_status(t_elapsed_s: float, device_states: dict):
    """
    Exibe no terminal o estado independente atualizado em tempo real de TODOS os dispositivos simultaneamente.
    """
    h = int(t_elapsed_s // 3600)
    m = int((t_elapsed_s % 3600) // 60)
    s = int(t_elapsed_s % 60)
    time_str = f"{h:02d}:{m:02d}:{s:02d}"

    print(f"\n[ MONITORAMENTO INDEPENDENTE MPPT | Tempo Decorrido: {time_str} ]" + "─" * 25)
    for dev_id, state in device_states.items():
        name = state["custom_name"]
        raw = state["raw_records"][-1] if state["raw_records"] else {}
        v_set = state["v_now"]
        v_m = raw.get("v_meas_V", 0.0)
        i_m = raw.get("i_meas_mA", 0.0)
        p_dens = raw.get("power_density_mw_cm2", 0.0)
        pce = raw.get("pce_percent", 0.0)
        t_dwell = raw.get("t_dwell_s", 0.0)
        steps = state["step_count"]

        phase = "REFINO" if not state["is_exploring"] else "EXPLORAR"
        print(f"  ├─ [{name:<20s}] Ciclos:{steps:4d} | V_set:{v_set:6.3f}V | V_meas:{v_m:6.3f}V | "
              f"I:{i_m:8.3f}mA | P:{p_dens:7.2f}mW/cm² | PCE:{pce:6.2f}% | t_dwell:{t_dwell:5.2f}s | Modo:{phase}")


def save_plots_and_reports(dev_state: dict):
    """
    Generate and save evolution plots and updated CSV reports for a single device (Channel A).
    """
    dev_dir = dev_state["output_dir"]
    raw_df = pd.DataFrame(dev_state["raw_records"])
    trans_df = pd.DataFrame(dev_state["transient_records"])

    # 1. Save CSVs
    raw_path = dev_dir / "MPPT_raw.csv"
    trans_path = dev_dir / "transient_data.csv"

    raw_df.to_csv(raw_path, index=False)
    if not trans_df.empty:
        trans_df.to_csv(trans_path, index=False)

    # 2. Power Evolution Plot
    if not raw_df.empty and "t_rel_h" in raw_df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(raw_df["t_rel_h"], raw_df["power_density_mw_cm2"], label="Potência (mW/cm²)", color="#1f77b4")
        ax.set_xlabel("Tempo (horas)")
        ax.set_ylabel("Potência (mW/cm²)")
        ax.set_title(f"Evolução de Potência - {dev_state['custom_name']}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dev_dir / "power_evolution.png", dpi=300)
        plt.close(fig)

        # 3. Voltage & Current Evolution Plot
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        ax1.plot(raw_df["t_rel_h"], raw_df["v_meas_V"], color="#2ca02c", label="Tensão Medida (V)", linewidth=1.2)
        ax2.plot(raw_df["t_rel_h"], raw_df["i_meas_mA"], color="#d62728", label="Corrente (mA)", linewidth=1.2)

        ax1.set_xlabel("Tempo (horas)")
        ax1.set_ylabel("Tensão (V)", color="#2ca02c")
        ax2.set_ylabel("Corrente (mA)", color="#d62728")
        ax1.set_title(f"Evolução de V e I - {dev_state['custom_name']}")
        fig.tight_layout()
        fig.savefig(dev_dir / "voltage_current_evolution.png", dpi=300)
        plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 75)
    print("  PROTOCOLO DE ESTABILIDADE TRANSIENTE LONG-TERM (MPPT INDEPENDENTE)")
    print("=" * 75)

    # 1. Discover and connect ADALM1000 devices
    driver = MultiADALM1000_Driver()
    try:
        driver.connect(curr_sour=False)
    except Exception as exc:
        print(f"Erro ao conectar ADALM1000: {exc}")
        return

    # 2. Interactive Hardware Identification Wizard
    device_states = {}
    print("\n" + "━" * 75)
    print("  ASSISTENTE DE IDENTIFICAÇÃO FÍSICA DOS DISPOSITIVOS (CANAL A)")
    print("━" * 75)
    print("O programa irá pulsar 2.0 V no Canal A de cada ADALM sequencialmente")
    print("para permitir identificar visualmente/multímetro a qual amostra ele corresponde.\n")

    dev_items = list(driver.devices.items())
    for idx, (dev_id, drv) in enumerate(dev_items):
        print(f"[{idx + 1}/{len(dev_items)}] Pulsando 2.0V no Canal A do Dispositivo [{dev_id}] por 2 segundos...")

        # Pulse 2.0 V on Channel A
        drv.set_voltage_and_measure(2.0)
        time.sleep(2.0)
        drv.set_voltage_and_measure(0.0)  # Return to 0V

        default_name = f"Dispositivo_{idx + 1}_{dev_id}"
        try:
            custom_name = input(f"   -> Digite o nome/etiqueta para este dispositivo (ex: Celula_1) [Enter para '{default_name}']: ").strip()
        except EOFError:
            custom_name = ""

        if not custom_name:
            custom_name = default_name

        dev_dir = BASE_OUTPUT_DIR / custom_name
        dev_dir.mkdir(parents=True, exist_ok=True)

        device_states[dev_id] = {
            "dev_id": dev_id,
            "drv": drv,
            "custom_name": custom_name,
            "output_dir": dev_dir,
            "v_now": V_START,
            "direction": 1,
            "is_exploring": True,
            "p_prev": 0.0,
            "step_count": 0,
            "window_start": time.perf_counter(),
            "window_t": [],
            "window_v": [],
            "window_i": [],
            "cv_buffer": [],
            "raw_records": [],
            "transient_records": [],
        }
        print(f"   ✓ Configurado: '{custom_name}' -> Pasta: '{dev_dir}'\n")

    print("Identificação finalizada.")
    print("Iniciando protocolo de estabilidade de longa duração (MPPT Independente por Dispositivo)...")
    print(f"Dispositivos configurados: {[st['custom_name'] for st in device_states.values()]}")

    t_start = time.perf_counter()
    last_checkpoint_h = 0  # Starts at 0: first transient analysis runs ONLY after 1h (hour 1)

    try:
        while (time.perf_counter() - t_start) < T_TOTAL:
            t_elapsed = time.perf_counter() - t_start
            current_hour = int(t_elapsed // HOURLY_CHECK_INTERVAL)

            # ── MPPT INDEPENDENTE POR DISPOSITIVO (NON-BLOCKING SCHEDULER) ───
            # Cada dispositivo amostra e gerencia sua própria janela CV e dwell time!
            status_changed = False

            for dev_id, state in device_states.items():
                drv = state["drv"]
                v_set = state["v_now"]

                # Amostra o Canal A deste dispositivo no seu setpoint atual
                v_m, i_m = drv.set_voltage_and_measure(v_set)
                t_w = time.perf_counter() - state["window_start"]

                state["window_t"].append(t_w)
                state["window_v"].append(v_m)
                state["window_i"].append(i_m)
                state["cv_buffer"].append(i_m)

                # Verifica o critério CV deste dispositivo específico
                is_settled = False
                if len(state["cv_buffer"]) > CV_WINDOW:
                    state["cv_buffer"].pop(0)
                    m = np.mean(state["cv_buffer"])
                    s = np.std(state["cv_buffer"])
                    if m != 0 and (s / abs(m)) < MIN_CV:
                        is_settled = True

                # Se este dispositivo estabilizou no CV OU estourou o TIMEOUT:
                if is_settled or t_w >= TIMEOUT_CV:
                    v_arr = np.array(state["window_v"])
                    i_arr = np.array(state["window_i"])

                    # Média estacionária dos últimos 30% da janela
                    n_pts = len(v_arr)
                    ss_start = int(0.7 * n_pts) if n_pts >= 5 else 0
                    v_ss = float(np.mean(v_arr[ss_start:])) if n_pts > 0 else v_m
                    i_ss = float(np.mean(i_arr[ss_start:])) if n_pts > 0 else i_m

                    power_mw = abs(v_ss * i_ss * 1e3)
                    power_density = (power_mw / SAMPLE_AREA)
                    pce = (power_density / P_IN) * 100.0

                    state["raw_records"].append({
                        "time_s": t_elapsed,
                        "t_rel_h": t_elapsed / 3600.0,
                        "v_set_V": v_set,
                        "v_meas_V": v_ss,
                        "i_meas_mA": i_ss * 1e3,
                        "power_mw": power_mw,
                        "power_density_mw_cm2": power_density,
                        "pce_percent": pce,
                        "t_dwell_s": t_w,
                    })

                    # Avança o algoritmo P&O deste dispositivo individualmente
                    v_next, new_dir, is_exp = po_step(
                        p_now=power_mw,
                        v_now=state["v_now"],
                        direction=state["direction"],
                        is_exploring=state["is_exploring"],
                        p_prev=state["p_prev"],
                    )

                    state["v_now"] = v_next
                    state["direction"] = new_dir
                    state["is_exploring"] = is_exp
                    state["p_prev"] = power_mw
                    state["step_count"] += 1
                    status_changed = True

                    # Reinicia os buffers de janela deste dispositivo
                    state["window_start"] = time.perf_counter()
                    state["window_t"].clear()
                    state["window_v"].clear()
                    state["window_i"].clear()
                    state["cv_buffer"].clear()

            # Exibe o status atualizado de todos os dispositivos no terminal quando houver avanço de ciclo
            if status_changed:
                print_mppt_status(t_elapsed, device_states)

            # ── HOURLY TRANSIENT ANALYSIS CHECKPOINT ─────────────────────────
            # Começa SOMENTE após completar pelo menos 1 hora do início da estabilidade (current_hour >= 1)
            if current_hour > last_checkpoint_h and current_hour >= 1:
                last_checkpoint_h = current_hour
                print(f"\n[{time.strftime('%H:%M:%S')}] >>> Iniciando Análise de Transiente de 1h (Hora {current_hour})...")

                for cyc in range(1, N_TRANSIENT_CYCLES + 1):
                    step_dir = 1 if (cyc % 2 != 0) else -1
                    t_v_targets = {}

                    for dev_id, state in device_states.items():
                        v_curr = state["v_now"]
                        v_step = float(np.clip(v_curr + step_dir * LARGE_STEP, 0.0, 1.3))
                        t_v_targets[dev_id] = v_step

                    # Acquire Channel A step responses until CV steady state
                    trans_res, _ = acquire_cv_transient_cha(driver, t_v_targets, timeout=TIMEOUT_CV)

                    # Fit double exponential model for each device Channel A
                    for dev_id, (t_arr, v_arr, i_arr) in trans_res.items():
                        state = device_states[dev_id]
                        v_before = state["v_now"]
                        v_after = t_v_targets[dev_id]
                        dt_cv = t_arr[-1] - t_arr[0] if len(t_arr) > 1 else 0.0

                        fit_res = double_exp_fitting(t_arr, i_arr)
                        a1, t1, a2, t2, c, r2 = fit_res

                        state["transient_records"].append({
                            "hour": current_hour,
                            "cycle_num": cyc,
                            "step_dir": step_dir,
                            "v_before_V": v_before,
                            "v_after_V": v_after,
                            "fit_a1": a1,
                            "fit_t1": t1,
                            "fit_a2": a2,
                            "fit_t2": t2,
                            "fit_c_steady_A": c,
                            "fit_r2": r2,
                            "cycle_duration_s": dt_cv,
                        })

                        # Plot transient fitting for hour H (first cycle)
                        if cyc == 1 and len(t_arr) > 5 and fit_res[0] is not None:
                            fig, ax = plt.subplots(figsize=(7, 4.5))
                            ax.plot(t_arr, i_arr * 1e3, "o", markersize=3, label="Medido (mA)", color="#1f77b4")
                            t_fit = np.linspace(t_arr[0], t_arr[-1], 200)
                            i_fit = (a1 * np.exp(-t_fit / t1) + a2 * np.exp(-t_fit / t2) + c) * 1e3
                            ax.plot(t_fit, i_fit, "--", label=f"Fit (R²={r2:.3f})", color="black")
                            ax.set_xlabel("Tempo (s)")
                            ax.set_ylabel("Corrente (mA)")
                            ax.set_title(f"Transiente Hora {current_hour} - {state['custom_name']}")
                            ax.legend()
                            fig.tight_layout()
                            fig.savefig(state["output_dir"] / f"transient_fits_hour_{current_hour}.png", dpi=300)
                            plt.close(fig)

                # Save CSVs and plots for all devices
                for state in device_states.values():
                    save_plots_and_reports(state)

                print(f" [OK] Análise de Transiente da Hora {current_hour} concluída e salva com sucesso.")

                # Reset window timers for all devices after transient checkpoint
                for state in device_states.values():
                    state["window_start"] = time.perf_counter()
                    state["window_t"].clear()
                    state["window_v"].clear()
                    state["window_i"].clear()
                    state["cv_buffer"].clear()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
    finally:
        print("\nFinalizando e salvando dados finais...")
        for state in device_states.values():
            save_plots_and_reports(state)
        driver.disconnect()
        print("Protocolo finalizado. Dispositivos desconectados.")


if __name__ == "__main__":
    main()