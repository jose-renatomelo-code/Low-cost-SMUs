import logging
import resource
import signal
import sys
import time
import traceback
import faulthandler
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

from drivers import MultiADALM1000_Driver, ADALM1000_Driver

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION & PARAMETERS (Apenas Canal A por ADALM1000)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V_START               = 0.6          # Initial tracking voltage default (V)
LARGE_STEP            = 0.1          # Exploration / transient step size (V)
SMALL_STEP            = 0.05         # Refinement step size (V)
SAMPLE_AREA           = 0.16         # Active area (cm²)
P_IN                  = 100.0        # Incident light intensity (mW/cm²)
T_TOTAL               = 7 * 24 * 3600  # Total duration (seconds): 1 week
HOURLY_CHECK_INTERVAL = 30         # Interval for transient analysis checkpoint (seconds: 3600 = 1h)
N_TRANSIENT_CYCLES    = 10           # Number of transient cycles per hourly evaluation
TIMEOUT_CV            = 15.0         # Max wait time for CV stabilization per cycle (seconds)
CV_WINDOW             = 20           # Sliding window size for CV check
MIN_CV                = 0.05         # Min CV threshold (5%)
MIN_CYCLE_TIME        = 1.0          # Minimum time between cycle starts (s) — avoid empty loops
HEARTBEAT_EVERY_N_CYCLES = 20        # How often to log a "still alive" heartbeat

BASE_OUTPUT_DIR = Path("OUTPUT STABILITY")
BASE_OUTPUT_DIR.mkdir(exist_ok=True)

# Tensão de partida por dispositivo (permite aplicar tensões diferentes desde o início).
# Chave: custom_name digitado no assistente. Dispositivos não listados usam V_START.
V_START_PER_DEVICE = {
    # "Celula_1": 0.60,
    # "Celula_2": 0.75,
    # "Celula_3": 0.90,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIAGNOSTICS / LOGGING SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fully-buffered stdout (the default whenever output is redirected to a file
# instead of a terminal) means print()'d status lines can sit in a buffer and
# never reach disk if the process dies abruptly. Force line buffering so the
# last thing printed before a crash is actually on disk.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

LOG_PATH = BASE_OUTPUT_DIR / "run.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# If a fatal signal (segfault/abort) comes from native driver/libusb code,
# faulthandler dumps a C-level traceback to stderr before the process dies —
# otherwise a native crash leaves zero trace.
faulthandler.enable()


def _handle_term_signal(signum, frame):
    """Logs the origin of an external kill (SIGTERM/SIGHUP) before dying,
    so a dropped SSH session / systemd stop is distinguishable from a bug."""
    logger.error("Received signal %s — process is being terminated externally.", signum)
    logger.error("Stack at signal time:\n%s", "".join(traceback.format_stack(frame)))
    raise SystemExit(1)


signal.signal(signal.SIGTERM, _handle_term_signal)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _handle_term_signal)


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
# EXPONENTIAL FITTING & MATHEMATICAL MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def double_exp_func(t: np.ndarray, a1: float, tau1: float, a2: float, tau2: float, c: float) -> np.ndarray:
    """
    Bi-exponential transient model:
    I(t) = a1 * exp(-t / tau1) + a2 * exp(-t / tau2) + c
    """
    return a1 * np.exp(-t / np.maximum(tau1, 1e-9)) + a2 * np.exp(-t / np.maximum(tau2, 1e-9)) + c


def single_exp_func(t: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
    """
    Single exponential transient model:
    I(t) = a * exp(-t / tau) + c
    """
    return a * np.exp(-t / np.maximum(tau, 1e-9)) + c


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-18 else 1.0


def fit_transient_double_exp(t_data: np.ndarray, i_data: np.ndarray) -> tuple[float | None, ...]:
    """
    Fits double exponential decay: I(t) = a1*exp(-t/tau1) + a2*exp(-t/tau2) + c.
    Returns: (a1, tau1, a2, tau2, c, r2)
    """
    if len(t_data) < 6 or curve_fit is None:
        return None, None, None, None, None, None

    mask = ~(np.isnan(t_data) | np.isnan(i_data) | np.isinf(t_data) | np.isinf(i_data))
    t_clean = t_data[mask] - t_data[mask][0]  # relative time starting at 0
    i_clean = i_data[mask]

    if len(t_clean) < 6:
        return None, None, None, None, None, None

    c_init = float(np.median(i_clean[-max(3, len(i_clean) // 5):]))
    delta_i = float(i_clean[0] - c_init)
    dt_total = float(t_clean[-1] - t_clean[0]) if t_clean[-1] > t_clean[0] else 1.0

    p0 = [
        0.6 * delta_i,
        max(1e-4, 0.1 * dt_total),
        0.4 * delta_i,
        max(1e-3, 0.6 * dt_total),
        c_init
    ]
    bounds = (
        [-np.inf, 1e-6, -np.inf, 1e-6, -np.inf],
        [np.inf, 1e4, np.inf, 1e4, np.inf]
    )

    try:
        popt, _ = curve_fit(double_exp_func, t_clean, i_clean, p0=p0, bounds=bounds, maxfev=10000)
        a1, tau1, a2, tau2, c = popt
        # Ensure tau1 is the faster component (tau1 <= tau2)
        if tau1 > tau2:
            a1, a2 = a2, a1
            tau1, tau2 = tau2, tau1
        i_pred = double_exp_func(t_clean, a1, tau1, a2, tau2, c)
        r2 = calculate_r_squared(i_clean, i_pred)
        return float(a1), float(tau1), float(a2), float(tau2), float(c), float(r2)
    except Exception:
        logger.debug("Double-exp fit (first attempt) failed:\n%s", traceback.format_exc())
        try:
            popt, _ = curve_fit(double_exp_func, t_clean, i_clean, p0=p0, maxfev=5000)
            a1, tau1, a2, tau2, c = popt
            if tau1 > tau2:
                a1, a2 = a2, a1
                tau1, tau2 = tau2, tau1
            i_pred = double_exp_func(t_clean, a1, tau1, a2, tau2, c)
            r2 = calculate_r_squared(i_clean, i_pred)
            return float(a1), float(tau1), float(a2), float(tau2), float(c), float(r2)
        except Exception:
            logger.debug("Double-exp fit (fallback) failed:\n%s", traceback.format_exc())
            return None, None, None, None, None, None


def fit_transient_single_exp(t_data: np.ndarray, i_data: np.ndarray) -> tuple[float | None, ...]:
    """
    Fits single exponential decay: I(t) = a*exp(-t/tau) + c.
    Returns: (a, tau, c, r2)
    """
    if len(t_data) < 4 or curve_fit is None:
        return None, None, None, None

    mask = ~(np.isnan(t_data) | np.isnan(i_data) | np.isinf(t_data) | np.isinf(i_data))
    t_clean = t_data[mask] - t_data[mask][0]
    i_clean = i_data[mask]

    if len(t_clean) < 4:
        return None, None, None, None

    c_init = float(np.median(i_clean[-max(3, len(i_clean) // 5):]))
    delta_i = float(i_clean[0] - c_init)
    dt_total = float(t_clean[-1] - t_clean[0]) if t_clean[-1] > t_clean[0] else 1.0

    p0 = [delta_i, max(1e-4, 0.3 * dt_total), c_init]
    bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, 1e4, np.inf])

    try:
        popt, _ = curve_fit(single_exp_func, t_clean, i_clean, p0=p0, bounds=bounds, maxfev=5000)
        a, tau, c = popt
        i_pred = single_exp_func(t_clean, a, tau, c)
        r2 = calculate_r_squared(i_clean, i_pred)
        return float(a), float(tau), float(c), float(r2)
    except Exception:
        logger.debug("Single-exp fit (first attempt) failed:\n%s", traceback.format_exc())
        try:
            popt, _ = curve_fit(single_exp_func, t_clean, i_clean, p0=p0, maxfev=3000)
            a, tau, c = popt
            i_pred = single_exp_func(t_clean, a, tau, c)
            r2 = calculate_r_squared(i_clean, i_pred)
            return float(a), float(tau), float(c), float(r2)
        except Exception:
            logger.debug("Single-exp fit (fallback) failed:\n%s", traceback.format_exc())
            return None, None, None, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MPPT HELPER FUNCTIONS
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
) -> tuple[dict, float]:
    """
    Applies v_targets (dev_id -> v_target) on Channel A for all devices simultaneously and
    acquires samples in parallel until EVERY device reaches steady state (CV < MIN_CV).
    The LONGEST device dictates the window duration: the cycle only advances once all
    devices are settled (settled = True) or the timeout expires.

    Returns:
    - result_window: dict mapping dev_id -> (t_arr, v_arr, i_arr, i_ss, settled, dt_ss, cv_final)
        t_arr/v_arr/i_arr : full acquisition window (relative to window start)
        i_ss             : steady-state current average (A)
        settled          : True if CV < MIN_CV was reached (False = TIMEOUT for this device)
        dt_ss            : time spent in steady state after settling (s), 0 if not settled
        cv_final         : final CV value (%)
    - window_duration: total time this cycle took (s) — driven by the slowest device
    """
    start_t = time.perf_counter()
    dev_keys = list(driver.devices.keys())

    data = {dev_id: {"t": [], "v": [], "i": []} for dev_id in dev_keys}
    cv_buffers = {dev_id: [] for dev_id in dev_keys}
    settled = {dev_id: False for dev_id in dev_keys}
    settle_t = {dev_id: None for dev_id in dev_keys}  # instant each device reached steady state

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
                if m != 0 and (s / abs(m)) < MIN_CV and not settled[dev_id]:
                    settled[dev_id] = True
                    settle_t[dev_id] = time.perf_counter()

    result_window = {}
    for dev_id in dev_keys:
        t_arr = np.array(data[dev_id]["t"])
        v_arr = np.array(data[dev_id]["v"])
        i_arr = np.array(data[dev_id]["i"])

        if len(cv_buffers[dev_id]):
            i_ss = float(np.mean(cv_buffers[dev_id]))
            cv_val = float(np.std(cv_buffers[dev_id]) / abs(i_ss) * 100.0) if i_ss != 0 else 0.0
        else:
            i_ss = float(i_arr[-1]) if len(i_arr) else 0.0
            cv_val = 0.0

        dt_ss = (time.perf_counter() - settle_t[dev_id]) if settle_t[dev_id] is not None else 0.0
        result_window[dev_id] = (t_arr, v_arr, i_arr, i_ss, settled[dev_id], dt_ss, cv_val)

    window_duration = time.perf_counter() - start_t
    return result_window, window_duration


def print_mppt_status(t_elapsed_s: float, device_states: dict, cycle: int = 0):
    """
    Exibe no terminal o estado independente atualizado em tempo real de TODOS os dispositivos
    simultaneamente. Chamado APENAS quando o ciclo avança — ou seja, quando TODOS os
    dispositivos estão em steady-state (o mais longo dita o instante de impressão).
    """
    h = int(t_elapsed_s // 3600)
    m = int((t_elapsed_s % 3600) // 60)
    s = int(t_elapsed_s % 60)
    time_str = f"{h:02d}:{m:02d}:{s:02d}"

    print(f"\n[ CYCLE {cycle} changed | {time_str} | ALL STEADY-STATE ]" + "─" * 25)
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

        phase = "REFINE" if not state["is_exploring"] else "EXPLORE"
        print(f"  ├─ [{name:<20s}] Cycles:{steps:4d} | V_set:{v_set:6.3f}V | V_meas:{v_m:6.3f}V | "
              f"I:{i_m:8.3f}mA | P:{p_dens:7.2f}mW/cm² | PCE:{pce:6.2f}% | t_dwell:{t_dwell:5.2f}s | Mode:{phase}")


def save_plots_and_reports(dev_state: dict):
    """
    Generate and save evolution plots and updated CSV reports for a single device (Channel A).
    """
    try:
        dev_dir = dev_state["output_dir"]
        raw_df = pd.DataFrame(dev_state["raw_records"])
        trans_df = pd.DataFrame(dev_state["transient_records"])

        # 1. Save CSVs
        raw_path = dev_dir / "MPPT_raw.csv"
        trans_path = dev_dir / "transient_metrics.csv"
        trans_compat_path = dev_dir / "transient_data.csv"

        raw_df.to_csv(raw_path, index=False)
        if not trans_df.empty:
            trans_df.to_csv(trans_path, index=False)
            trans_df.to_csv(trans_compat_path, index=False)

        # 2. Power Evolution Plot
        if not raw_df.empty and "t_rel_h" in raw_df.columns:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(raw_df["t_rel_h"], raw_df["power_density_mw_cm2"], label="Power (mW/cm²)", color="#1f77b4")
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Power (mW/cm²)")
            ax.set_title(f"Power Evolution - {dev_state['custom_name']}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(dev_dir / "power_evolution.png", dpi=300)
            plt.close(fig)

            # 3. Voltage & Current Evolution Plot
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax2 = ax1.twinx()

            ax1.plot(raw_df["t_rel_h"], raw_df["v_meas_V"], color="#2ca02c", label="Measured Voltage (V)", linewidth=1.2)
            ax2.plot(raw_df["t_rel_h"], raw_df["i_meas_mA"], color="#d62728", label="Current (mA)", linewidth=1.2)

            ax1.set_xlabel("Time (hours)")
            ax1.set_ylabel("Voltage (V)", color="#2ca02c")
            ax2.set_ylabel("Current (mA)", color="#d62728")
            ax1.set_title(f"IV Evolution - {dev_state['custom_name']}")
            fig.tight_layout()
            fig.savefig(dev_dir / "voltage_current_evolution.png", dpi=300)
            plt.close(fig)
    except Exception:
        # Never let a plotting/IO failure take down the measurement loop —
        # log it and keep the run alive; data is still safe in memory.
        logger.error("save_plots_and_reports failed for device '%s':\n%s",
                      dev_state.get("custom_name", "?"), traceback.format_exc())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 75)
    print("  PROTOCOLO DE ESTABILIDADE TRANSIENTE LONG-TERM (MPPT INDEPENDENTE)")
    print("=" * 75)
    logger.info("Run starting. T_TOTAL=%.1fh", T_TOTAL / 3600)

    # 1. Discover and connect ADALM1000 devices
    driver = MultiADALM1000_Driver()
    try:
        driver.connect(curr_sour=False)
    except Exception as exc:
        print(f"Erro ao conectar ADALM1000: {exc}")
        logger.error("Failed to connect to ADALM1000 devices:\n%s", traceback.format_exc())
        return

    # 2. Interactive Hardware Identification Wizard
    device_states = {}
    print("\n" + "━" * 75)
    print("  ASSISTENTE DE IDENTIFICAÇÃO FÍSICA DOS DISPOSITIVOS (CANAL A)")
    print("━" * 75)
    print("O programa irá piscar os LEDs de cada ADALM1000 para identificação")

    dev_items = list(driver.devices.items())
    for idx, (dev_id, drv) in enumerate(dev_items):
        print(f"[{idx + 1}/{len(dev_items)}] Piscando o LED do [{dev_id}] 3 vezes...")

        # Piscando o LED
        drv.blink_led(on_off_time=1, n_blinks=3)

        default_name = f"Dispositivo_{idx + 1}_{dev_id}"
        try:
            custom_name = input(f"   -> Digite o nome/etiqueta para este dispositivo (ex: Celula_1) [Enter para '{default_name}']: ").strip()
        except EOFError:
            custom_name = ""

        if not custom_name:
            custom_name = default_name

        dev_dir = BASE_OUTPUT_DIR / custom_name
        dev_dir.mkdir(parents=True, exist_ok=True)

        v_init_default = float(V_START_PER_DEVICE.get(custom_name, V_START))
        try:
            v_input = input(f"   -> Tensão inicial de operação (V_start) [Enter para {v_init_default:.3f}V]: ").strip()
            v_start = float(v_input) if v_input else v_init_default
        except (ValueError, EOFError):
            v_start = v_init_default

        device_states[dev_id] = {
            "dev_id": dev_id,
            "drv": drv,
            "custom_name": custom_name,
            "output_dir": dev_dir,
            "v_now": v_start,
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
        print(f"   ✓ Configurado: '{custom_name}' | V_start={v_start:.3f}V -> Pasta: '{dev_dir}'\n")
        logger.info("Device configured: %s (dev_id=%s, v_start=%.3f)", custom_name, dev_id, v_start)

    print("Identificação finalizada.")
    print("Iniciando protocolo de estabilidade de longa duração (MPPT Paralelo Independente por Dispositivo)...")
    print(f"Dispositivos configurados: {[st['custom_name'] for st in device_states.values()]}")
    logger.info("All devices configured, entering main loop.")

    t_start = time.perf_counter()
    last_checkpoint_h = 0  # Starts at 0: first hourly transient runs after 1h (current_hour == 1)

    try:
        cycle = 0
        while (time.perf_counter() - t_start) < T_TOTAL:
            # ── 1) Aquisição da janela MPPT em paralelo:
            #      Aplica a tensão independente de cada dispositivo e amostra
            #      até TODOS atingirem steady-state (CV < MIN_CV) ou timeout.
            #      O dispositivo mais lento dita a duração da janela.
            cycle_t0 = time.perf_counter() - t_start
            v_targets = {dev_id: st["v_now"] for dev_id, st in device_states.items()}

            try:
                res, window_duration = acquire_cv_transient_cha(driver, v_targets, timeout=TIMEOUT_CV)
            except Exception:
                logger.error("acquire_cv_transient_cha failed in MPPT cycle %d:\n%s",
                              cycle, traceback.format_exc())
                time.sleep(2.0)
                continue

            cycle += 1
            t_elapsed = time.perf_counter() - t_start
            current_hour = int(t_elapsed // HOURLY_CHECK_INTERVAL)

            print(f"\n--- Ciclo {cycle} | janela de {window_duration:.2f}s "
                  f"(todos os dispositivos em steady-state) ---")

            for dev_id, (t_arr, v_arr, i_arr, i_ss, settled, dt_ss, cv_val) in res.items():
                try:
                    state = device_states[dev_id]
                    v_meas = float(np.mean(v_arr)) if len(v_arr) else float(state["v_now"])
                    power_mw = abs(v_meas * i_ss * 1000.0)  # mW total
                    power_density = power_mw / SAMPLE_AREA  # mW/cm²
                    pce = power_mw / (P_IN * SAMPLE_AREA) * 100.0  # %

                    state["raw_records"].append({
                        "time_s": t_elapsed,
                        "t_rel_h": t_elapsed / 3600.0,
                        "v_set_V": float(state["v_now"]),
                        "v_meas_V": v_meas,
                        "i_meas_mA": i_ss * 1e3,
                        "power_mw": power_mw,
                        "power_density_mw_cm2": power_density,
                        "pce_percent": pce,
                        "t_dwell_s": window_duration,
                        "settled": bool(settled),
                        "dt_ss_s": dt_ss,
                        "cv_percent": cv_val,
                    })

                    # Avança o algoritmo P&O deste dispositivo individualmente a partir do V real medido
                    v_next, new_dir, is_exp = po_step(
                        p_now=power_mw,
                        v_now=v_meas,
                        direction=state["direction"],
                        is_exploring=state["is_exploring"],
                        p_prev=state["p_prev"],
                    )

                    # Clamp de segurança ao range do instrumento
                    v_max = getattr(driver, "V_MAX_V", None) or 2.0
                    v_next = float(np.clip(v_next, 0.0, v_max))

                    state["v_now"] = v_next
                    state["direction"] = new_dir
                    state["is_exploring"] = is_exp
                    state["p_prev"] = power_mw
                    state["step_count"] += 1
                except Exception:
                    logger.error("Error processing device %s in MPPT cycle %d:\n%s",
                                  dev_id, cycle, traceback.format_exc())
                    continue

            # Exibe o status atualizado de todos os dispositivos no terminal
            print_mppt_status(t_elapsed, device_states, cycle=cycle)

            # Heartbeat: cheap proof-of-life + memory trend, independent of
            # everything else below. If the run dies, this tells you the
            # last confirmed-alive cycle and whether memory was climbing.
            if cycle % HEARTBEAT_EVERY_N_CYCLES == 0:
                mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                logger.info("Heartbeat: cycle=%d elapsed=%.2fh mem=%.1fMB devices=%d",
                            cycle, t_elapsed / 3600, mem_mb, len(device_states))

            # ── HOURLY TRANSIENT ANALYSIS CHECKPOINT ─────────────────────────
            # Dispara de hora em hora após o início (current_hour >= 1)
            if current_hour > last_checkpoint_h and current_hour >= 1:
                last_checkpoint_h = current_hour
                print(f"\n[{time.strftime('%H:%M:%S')}] >>> Iniciando Análise de Transiente Horária (Hora {current_hour})...")
                logger.info("Entering hourly transient checkpoint for hour %d", current_hour)

                try:
                    # Salva o estado normal de MPPT (tensão base) de cada dispositivo
                    v_mpp_baseline = {dev_id: state["v_now"] for dev_id, state in device_states.items()}
                    v_prev_applied = {dev_id: state["v_now"] for dev_id, state in device_states.items()}

                    for cyc in range(1, N_TRANSIENT_CYCLES + 1):
                        # Oscila a perturbação variando dir = +1 e -1 com LARGE_STEP em torno do MPPT
                        step_dir = 1 if (cyc % 2 != 0) else -1
                        v_transient_targets = {}

                        for dev_id, state in device_states.items():
                            v_base = v_mpp_baseline[dev_id]
                            v_step = float(np.clip(v_base + step_dir * LARGE_STEP, 0.0, 1.3))
                            v_transient_targets[dev_id] = v_step

                        # Aquisição paralela da resposta ao degrau em Canal A para todos os dispositivos
                        try:
                            trans_res, trans_duration = acquire_cv_transient_cha(driver, v_transient_targets, timeout=TIMEOUT_CV)
                        except Exception:
                            logger.error("acquire_cv_transient_cha failed in transient cycle %d/hour %d:\n%s",
                                         cyc, current_hour, traceback.format_exc())
                            continue

                        # Fitting exponencial completo e registro de métricas para cada dispositivo
                        for dev_id, (t_arr, v_arr, i_arr, i_ss, settled, dt_ss, cv_val) in trans_res.items():
                            try:
                                state = device_states[dev_id]
                                v_before = v_prev_applied[dev_id]
                                v_after = v_transient_targets[dev_id]
                                delta_v = v_after - v_before
                                dt_cycle = t_arr[-1] - t_arr[0] if len(t_arr) > 1 else 0.0
                                i_init_mA = float(i_arr[0] * 1e3) if len(i_arr) else 0.0
                                i_mean_20_mA = float(np.mean(i_arr[-20:]) * 1e3) if len(i_arr) >= 20 else float(np.mean(i_arr) * 1e3) if len(i_arr) else 0.0
                                t_rel_step = t_arr - t_arr[0] if len(t_arr) else t_arr

                                # 1. Fitting Bi-exponencial: I(t) = a1*exp(-t/t1) + a2*exp(-t/t2) + c
                                a1, tau1, a2, tau2, c_double, r2_double = fit_transient_double_exp(t_rel_step, i_arr)

                                # 2. Fitting Mono-exponencial: I(t) = a*exp(-t/tau) + c
                                a_single, tau_single, c_single, r2_single = fit_transient_single_exp(t_rel_step, i_arr)

                                # Armazena todas as métricas detalhadas
                                state["transient_records"].append({
                                    "timestamp_s": time.perf_counter() - t_start,
                                    "hour": current_hour,
                                    "cycle_num": cyc,
                                    "step_dir": step_dir,
                                    "v_before_V": v_before,
                                    "v_after_V": v_after,
                                    "delta_v_V": delta_v,
                                    "i_initial_mA": i_init_mA,
                                    "i_mean_20_mA": i_mean_20_mA,
                                    "i_ss_meas_mA": i_ss * 1e3,
                                    "cv_final_percent": cv_val,
                                    "settled": bool(settled),
                                    "settling_time_s": dt_cycle - dt_ss if dt_ss > 0 else dt_cycle,
                                    "cycle_duration_s": dt_cycle,
                                    # Parâmetros Bi-exponencial
                                    "fit_double_a1_mA": a1 * 1e3 if a1 is not None else None,
                                    "fit_double_tau1_s": tau1 if tau1 is not None else None,
                                    "fit_double_tau1_ms": tau1 * 1e3 if tau1 is not None else None,
                                    "fit_double_a2_mA": a2 * 1e3 if a2 is not None else None,
                                    "fit_double_tau2_s": tau2 if tau2 is not None else None,
                                    "fit_double_c_ss_mA": c_double * 1e3 if c_double is not None else None,
                                    "fit_double_r2": r2_double if r2_double is not None else None,
                                    # Parâmetros Mono-exponencial
                                    "fit_single_a_mA": a_single * 1e3 if a_single is not None else None,
                                    "fit_single_tau_s": tau_single if tau_single is not None else None,
                                    "fit_single_c_ss_mA": c_single * 1e3 if c_single is not None else None,
                                    "fit_single_r2": r2_single if r2_single is not None else None,
                                })

                                # Gera gráfico detalhado do fitting para o primeiro ciclo (ou quando houver bom ajuste)
                                if cyc == 1 and len(t_arr) > 5:
                                    fig, ax = plt.subplots(figsize=(8, 5))
                                    ax.plot(t_rel_step, i_arr * 1e3, "o", markersize=3, label="Data", color="#1f77b4", alpha=0.8)
                                    ax.legend(loc="upper right")

                                    if a1 is not None and tau1 is not None:
                                        t_fit = np.linspace(t_rel_step[0], t_rel_step[-1], 250)
                                        i_fit_double = double_exp_func(t_fit, a1, tau1, a2, tau2, c_double) * 1e3
                                        ax.plot(t_fit, i_fit_double, "-", label=f"Bi-Exp Fit (R²={r2_double:.3f})", color="#d62728", linewidth=2.0)

                                        param_text = (
                                            f"a₁ = {a1*1e3:.3f} mA\n"
                                            f"τ₁ = {tau1*1e3:.2f} ms\n"
                                            f"a₂ = {a2*1e3:.3f} mA\n"
                                            f"τ₂ = {tau2:.3f} s\n"
                                            f"I_ss(fit) = {c_double*1e3:.3f} mA\n"
                                            f"I_ss(med) = {i_ss*1e3:.3f} mA\n"
                                            f"R² = {r2_double:.4f}\n"
                                            f"CV = {cv_val:.2f}%\n"
                                            f"ΔV = {delta_v:+.2f}V"
                                        )
                                        ax.text(
                                            0.97, 0.95, param_text,
                                            transform=ax.transAxes,
                                            fontsize=9,
                                            verticalalignment="center",
                                            horizontalalignment="right",
                                            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#cccccc")
                                        )

                                    ax.set_xlabel("Relative Step Time (s)")
                                    ax.set_ylabel("Current (mA)")
                                    ax.set_title(f"Transient Hour {current_hour} (Cycle 1) - {state['custom_name']} (ΔV={delta_v:+.2f}V)")
                                    fig.tight_layout()
                                    fig.savefig(state["output_dir"] / f"transient_fits_hour_{current_hour}.png", dpi=300)
                                    plt.close(fig)
                            except Exception:
                                logger.error("Error processing transient result for device %s, hour %d, cycle %d:\n%s",
                                              dev_id, current_hour, cyc, traceback.format_exc())
                                continue

                        # Atualiza a tensão previamente aplicada para o próximo degrau
                        for dev_id, v_target in v_transient_targets.items():
                            v_prev_applied[dev_id] = v_target

                    # ── RESTAURAÇÃO: Retorna todos os dispositivos à tensão de MPPT baseline
                    print(f" >>> Restaurando dispositivos para as tensões de MPPT de operação normal...")
                    driver.set_voltage_channel_a_all_and_measure(v_mpp_baseline)
                    for dev_id, state in device_states.items():
                        state["v_now"] = v_mpp_baseline[dev_id]

                    # Salva CSVs e gráficos atualizados para todos os dispositivos
                    for state in device_states.values():
                        save_plots_and_reports(state)

                    print(f" [OK] Análise de Transiente da Hora {current_hour} concluída e dados salvos com sucesso.\n")
                    logger.info("Hourly transient checkpoint for hour %d completed OK.", current_hour)

                    # Reinicia os buffers de janela dos dispositivos após o checkpoint de transiente
                    for state in device_states.values():
                        state["window_start"] = time.perf_counter()
                        state["window_t"].clear()
                        state["window_v"].clear()
                        state["window_i"].clear()
                        state["cv_buffer"].clear()
                except Exception:
                    # An error anywhere in the hourly checkpoint (voltage
                    # excursion, driver call, curve fitting) must not kill
                    # the whole week-long run — log it, restore baseline
                    # voltages defensively, and keep going.
                    logger.error("Hourly transient checkpoint for hour %d failed:\n%s",
                                  current_hour, traceback.format_exc())
                    try:
                        driver.set_voltage_channel_a_all_and_measure(
                            {dev_id: state["v_now"] for dev_id, state in device_states.items()}
                        )
                    except Exception:
                        logger.error("Failed to restore baseline voltages after checkpoint error:\n%s",
                                      traceback.format_exc())

            # Garante um tempo mínimo de ciclo (evita loop vazio em hardware rápido)
            cycle_dt = time.perf_counter() - cycle_t0
            if cycle_dt < MIN_CYCLE_TIME:
                time.sleep(MIN_CYCLE_TIME - cycle_dt)

    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
        logger.info("Interrupted by user (KeyboardInterrupt) at cycle %d.", cycle)
    except Exception:
        logger.error("Fatal error, aborting run at cycle %d:\n%s", cycle, traceback.format_exc())
        raise
    finally:
        print("\nFinalizando e salvando dados finais...")
        for state in device_states.values():
            save_plots_and_reports(state)
        driver.disconnect()
        print("Protocolo finalizado. Dispositivos desconectados.")
        logger.info("Run finished, devices disconnected.")


if __name__ == "__main__":
    main()