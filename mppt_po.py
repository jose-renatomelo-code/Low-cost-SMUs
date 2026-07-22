from drivers import (ADALM1000_Driver, AD3_Driver,
                    Keithley2450_Driver, USMU_Driver)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from jv_sweep import build_driver, preconditioning_device
import time

try:
    from scipy.optimize import curve_fit
except ImportError:  # scipy opcional: só necessário no método "fitting"
    curve_fit = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURAÇÃO  –  INSTRUMENTO, MÉTODO E PARÂMETROS DO MPPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUMENT = "USMU"   # "USMU" | "KEITHLEY" | "ADALM1000" | "AD3"
OUTPUT_DIR  = Path("output MPPT")
OUTPUT_DIR.mkdir(exist_ok=True)

# Método de aquisição / determinação do estado estacionário (steady-state):
#   "cv"      -> amostra até o coeficiente de variação (CV) da corrente baixar
#                abaixo de MIN_CV (critério de estabilidade)
#   "fixed"   -> aguarda um tempo fixo de dwell (T_DWELL) e usa a média
#   "fitting" -> aguarda T_DWELL e extrapola a corrente de estado estacionário
#                via ajuste bi-exponencial da transiente
METHOD = "fitting"               # "cv" / "fitting" / "fixed"
CV_WINDOW = 20              # nº de amostras na janela deslizante do critério CV
MIN_CV = 0.01              # CV máximo (0.1%) para considerar estado estacionário

# Orientação do passo inicial do algoritmo Perturb & Observe:
#   "FORWARD"    -> sobe a tensão (direction = +1)
#   "REVERSE"    -> desce a tensão  (direction = -1)
#   "FROM_VOC"   -> faz pré-condicionamento, parte de Voc e desce (direction = -1)
ORIENTATION = "FORWARD"     # "FORWARD" / "REVERSE" / "FROM_VOC"

# Abordagem de controle. GALVANOSTATIC ainda não está implementado.
APPROACH = "POTENTIOSTATIC" # "POTENTIOSTATIC" / "GALVANOSTATIC"

# ── Parâmetros do MPPT (Perturb & Observe) ───────────────────────────────────
V_START    = 0.5     # V  – tensão inicial do rastreamento
T_DWELL    = 2       # s  – tempo de dwell (métodos "fixed" / "fitting")
LARGE_STEP = 0.1     # V  – passo de perturbação na fase de EXPLORAÇÃO
SMALL_STEP = 0.05    # V  – passo de perturbação na fase de REFINO (perto do MPP)
SAMPLE_AREA = 5      # cm²
P_IN       = 100     # mW/cm²  – irradiância incidente (para o cálculo de PCE)
T_TOTAL    = 60      # s  – duração total do rastreamamento
TIMEOUT    = 15      # s  – tempo máximo de espera por estado estacionário (método "cv")
MIN_CYCLE_TIME = 0.05  # s  – tempo mínimo por ciclo (evita loop vazio em hardware rápido)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ESTILO DE PLOT (igual ao jv_sweep.py, para manter consistência visual)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
# AJUSTE BI-EXPONENCIAL DA TRANSIENTE (método "fitting")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def double_exp_func(t, a1, a2, t1, t2, c):
    """Soma de duas exponenciais decrescentes + offset (estado estacionário)."""
    return a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2) + c


def calculate_r_squared(y_true, y_pred):
    """Coeficiente de determinação R² do ajuste."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan


def double_exp_fitting(xdata, ydata):
    """Ajusta a função bi-exponencial aos dados.

    Retorna [a1, t1, a2, t2, c, R²]. Em caso de erro retorna uma lista de None.
    A corrente de estado estacionário é o offset 'c'.
    """
    # Remove NaN / Inf
    mask = ~(np.isnan(ydata) | np.isinf(ydata))
    xdata_clean = np.delete(xdata, np.where(~mask)[0])
    ydata_clean = np.delete(ydata, np.where(~mask)[0])

    # scipy é opcional — sem ele não dá para ajustar
    if curve_fit is None:
        print("[fitting] scipy não disponível — instale scipy ou use METHOD='cv'/'fixed'.")
        return [None, None, None, None, None, None]

    # Precisa de pontos suficientes para 5 parâmetros livres
    if len(ydata_clean) < 6:
        return [None, None, None, None, None, None]

    init = [1e-3, 50e-3, 1e-5, 1, 3e-3]
    try:
        params, _ = curve_fit(double_exp_func, xdata_clean, ydata_clean,
                              p0=init, maxfev=5000)
        y_fitted = double_exp_func(xdata_clean, *params)
        r_squared = calculate_r_squared(ydata_clean, y_fitted)
        return list(params) + [r_squared]
    except Exception as e:
        print(f"[fitting] erro no ajuste bi-exponencial: {e}")
        return [None, None, None, None, None, None]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AQUISIÇÃO DE UMA JANELA (um passo do Perturb & Observe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def acquire_cv_window(driver, v_now, cycle_start_time):
    """Aplica V=v_now e amostra até a corrente estabilizar (CV < MIN_CV) ou TIMEOUT.

    A janela é cronometrada a partir de seu próprio início (window_start),
    não do início do tracking — o TIMEOUT e o t_rel referem-se ao ciclo atual.

    Retorna (t, v, i) como arrays numpy (tempo relativo ao início da janela).
    """
    t_buf, v_buf, i_buf = [], [], []
    steady_state = False
    cv_buf = []
    window_start = time.perf_counter()
    while (not steady_state) and (time.perf_counter() - window_start) < TIMEOUT:
        vv, ii = driver.set_voltage_and_measure(v_now)
        t_rel = time.perf_counter() - window_start
        t_buf.append(t_rel)
        v_buf.append(vv)
        i_buf.append(ii)

        cv_buf.append(ii)
        if len(cv_buf) > CV_WINDOW:
            del cv_buf[0]
            i_mean = np.mean(cv_buf)
            i_std = np.std(cv_buf)
            if i_mean != 0 and abs(i_std / i_mean) < MIN_CV:
                steady_state = True


    if not steady_state:
        print("[cv] TIMEOUT atingido sem estabilidade — usando janela completa.")
    return np.array(t_buf), np.array(v_buf), np.array(i_buf)


def acquire_dwell_window(driver, v_now, cycle_start_time, dwell):
    """Aplica V=v_now e amostra por 'dwell' segundos (métodos fixed/fitting).

    A janela é cronometrada a partir de seu próprio início (window_start),
    não do início do tracking — caso contrário ciclos posteriores adquirem
    janelas de duração ~0 (bug: o while usava cycle_start_time como referência).

    Retorna (t, v, i) como arrays numpy (tempo relativo ao início da janela).
    """
    t_buf, v_buf, i_buf = [], [], []
    window_start = time.perf_counter()
    while (time.perf_counter() - window_start) < dwell:
        vv, ii = driver.set_voltage_and_measure(v_now)
        t_rel = time.perf_counter() - window_start
        t_buf.append(t_rel)
        v_buf.append(vv)
        i_buf.append(ii)
    return np.array(t_buf), np.array(v_buf), np.array(i_buf)


def process_window(t, v, i, method):
    """Processa a janela adquirida e devolve a corrente média / estacionária.

    - "cv" / "fixed": média aritmética da corrente da janela.
    - "fitting":     offset 'c' do ajuste bi-exponencial (estado estacionário
                     extrapolado); cai para a média se o ajuste falhar.

    Retorna (avg_v, avg_i_A, info) onde info traz detalhes do ajuste.
    """
    avg_v = float(v[-1]) if len(v) else np.nan

    if method == "fitting":
        # Tempo relativo ao início da janela (a transiente parte de t=0)
        t_rel = t - t[0] if len(t) else t
        params = double_exp_fitting(t_rel, i)
        c = params[4]
        if c is not None and np.isfinite(c):
            avg_i = float(c)
            info = {"fit_a1": params[0], "fit_t1": params[1],
                    "fit_a2": params[2], "fit_t2": params[3],
                    "fit_c": params[4], "fit_r2": params[5]}
        else:
            avg_i = float(np.mean(i)) if len(i) else np.nan
            info = {"fit_c": None, "fit_r2": None}
    else:
        avg_i = float(np.mean(i)) if len(i) else np.nan
        info = {}

    return avg_v, avg_i, info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LÓGICA PERTURB & OBSERVE (PO)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def po_logic(p_now, v_avg, direction, is_exploring, p_prev):
    """Decide o próximo passo do rastreamento MPPT.

    Parâmetros
    ----------
    p_now         : potência média desta janela (mW, valor absoluto)
    v_avg         : tensão média desta janela (V)
    direction     : direção atual do passo (+1 sobe V, -1 desce V)
    is_exploring  : True se ainda procura a região do MPP (passo grande)
    p_prev        : potência da janela anterior (mW, abs); 0 no 1º ciclo

    Retorna
    -------
    (v_next, direction, is_exploring, step)
    """
    # Primeiro ciclo: não há referência anterior, mantém a direção.
    if p_prev != 0:
        dP = p_now - p_prev          # >0: potência subiu (boa direção)
        if dP < 0:
            direction *= -1          # potência caiu -> inverte a direção
            if is_exploring:
                # Acabou de cruzar o MPP: entra em modo de REFINO (passo menor)
                is_exploring = False
                print("INFO: direção invertida — região do MPP encontrada (REFINO).")

    step = LARGE_STEP if is_exploring else SMALL_STEP
    v_next = v_avg + step * direction
    return v_next, direction, is_exploring, step


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT E SAVE (salva os dados brutos + resumo do tracking + figuras)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_and_save(raw_df, po_df, mpp, smu_dir, instrument, method):
    """Gera CSVs e figuras científicas do rastreamento MPPT.

    Figuras:
      1 - tensão bruta vs tempo
      2 - corrente bruta vs tempo
      3 - PCE (ou densidade de potência) por ciclo  -> convergência ao MPP
      4 - evolução do dwell (potência vs tempo)      -> só se method != "fixed"

    CSVs:
      mppt_raw.csv         -> todas as amostras brutas
      mppt_tracking.csv    -> um ponto por ciclo PO
      mppt_metrics.csv     -> resumo (MPP encontrado, PCE máx, etc.)
    """
    smu_dir.mkdir(parents=True, exist_ok=True)

    # ── CSVs ───────────────────────────────────────────────────────────────
    raw_path = smu_dir / "mppt_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw CSV: {raw_path}")

    track_path = smu_dir / "mppt_tracking.csv"
    po_df.to_csv(track_path, index=False)
    print(f"Saved tracking CSV: {track_path}")

    metrics = {
        "Instrument": instrument,
        "Method": method,
        "Orientation": ORIENTATION,
        "Approach": APPROACH,
        "SampleArea(cm²)": SAMPLE_AREA,
        "P_in(mW/cm²)": P_IN,
        "T_total(s)": T_TOTAL,
        "V_MPP(V)": mpp["v"],
        "J_MPP(mA/cm²)": mpp["j"],
        "P_MPP(mW)": mpp["p_mW"],
        "PCE_max(%)": mpp["pce"],
        "N_cycles": len(po_df),
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_path = smu_dir / "mppt_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics CSV: {metrics_path}")

    # ── Fig 1: tensão bruta vs tempo ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(raw_df["time(s)"], raw_df["voltage(V)"], color="#1f77b4", lw=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"MPPT Raw Voltage — {instrument} [{method}]")
    fig.tight_layout()
    fig.savefig(smu_dir / "mppt_raw_voltage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {smu_dir / 'mppt_raw_voltage.png'}")

    # ── Fig 2: corrente bruta vs tempo ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(raw_df["time(s)"], raw_df["j_current(mA/cm²)"], color="#ff7f0e", lw=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current Density J (mA/cm²)")
    ax.set_title(f"MPPT Raw Current — {instrument} [{method}]")
    fig.tight_layout()
    fig.savefig(smu_dir / "mppt_raw_current.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {smu_dir / 'mppt_raw_current.png'}")

    # ── Fig 3: evolução do tracking (PCE por ciclo) ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2ca02c" if m == "REFINE" else "#1f77b4"
              for m in po_df["mode"]]
    ax.scatter(po_df["cycle"], po_df["PCE(%)"], c=colors, s=30, zorder=3)
    # linha conectando os pontos na ordem do ciclo
    ax.plot(po_df["cycle"], po_df["PCE(%)"], color="#999999", lw=0.8, zorder=2)
    if np.isfinite(mpp["pce"]):
        ax.axhline(mpp["pce"], color="#d62728", ls="--", lw=1.5,
                   label=f"PCE MPP = {mpp['pce']:.2f}%")
    ax.set_xlabel("PO Cycle")
    ax.set_ylabel("PCE (%)")
    ax.set_title(f"MPPT Tracking Convergence — {instrument} [{method}]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(smu_dir / "mppt_tracking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {smu_dir / 'mppt_tracking.png'}")

    # ── Fig 4: evolução do dwell (potência vs tempo) — só p/ métodos não-fixed
    if method != "fixed":
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(raw_df["time(s)"], raw_df["power_density(mW/cm²)"],
                color="#9467bd", lw=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power Density (mW/cm²)")
        ax.set_title(f"MPPT Power Evolution (dwell) — {instrument} [{method}]")
        fig.tight_layout()
        fig.savefig(smu_dir / "mppt_dwell.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {smu_dir / 'mppt_dwell.png'}")
    else:
        print("INFO: método 'fixed' — figura de dwell ignorada (3 figuras).")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    # GALVANOSTATIC ainda não implementado — trava cedo com mensagem clara.
    if APPROACH == "GALVANOSTATIC":
        raise NotImplementedError(
            "Abordagem GALVANOSTATIC ainda não implementada. "
            "Use APPROACH = 'POTENTIOSTATIC'."
        )

    print(f"\nConnecting to instrument: {INSTRUMENT}")
    driver = build_driver(INSTRUMENT)
    smu_dir = OUTPUT_DIR / INSTRUMENT / METHOD
    smu_dir.mkdir(parents=True, exist_ok=True)

    # Conecta em modo fonte de tensão (potenciostático)
    driver.connect(curr_sour=False)

    # FROM_VOC: pré-condiciona e parte do Voc medido
    if ORIENTATION == "FROM_VOC":
        if hasattr(driver, "set_source_mode"):
            driver.set_source_mode(curr_sour=True)
        try:
            voc = preconditioning_device(driver)
            v_start = voc
            print(f"FROM_VOC: iniciando rastreamento a partir de Voc = {v_start:.4f} V")
        except Exception as e:
            v_start = V_START
            print(f"FROM_VOC preconditioning falhou ({e}); usando V_START={v_start} V")
        if hasattr(driver, "set_source_mode"):
            driver.set_source_mode(curr_sour=False)
    else:
        v_start = V_START

    # Configura integração / NPLC (após voltar ao modo tensão)
    if INSTRUMENT == "USMU":
        driver.configure_integration(1)
    elif INSTRUMENT == "KEITHLEY":
        driver.configure_integration(1)

    # Direção inicial conforme a orientação escolhida
    direction = 1 if ORIENTATION == "FORWARD" else -1

    # Estado do algoritmo
    is_exploring = True       # True=EXPLORAÇÃO (passo grande), False=REFINO
    v_now = v_start
    p_prev = 0.0              # potência da janela anterior (abs, mW)

    # Melhor ponto de potência máxima (MPP) encontrado até agora
    mpp = {"v": np.nan, "j": np.nan, "p_mW": -np.inf, "pce": np.nan, "cycle": -1}

    # Acumuladores
    raw_records = []          # uma linha por amostra bruta
    po_records = []           # uma linha por ciclo PO

    print("Starting MPPT tracking...")
    cycle_start_time = time.perf_counter()
    cycle = 0
    try:
        while (time.perf_counter() - cycle_start_time) < T_TOTAL:
            cycle_t0 = time.perf_counter()
            # ── 1) Aquisição da janela ──────────────────────────────────────
            if METHOD == "cv":
                t_win, v_win, i_win = acquire_cv_window(driver, v_now, cycle_start_time)
            else:  # "fixed" ou "fitting"
                t_win, v_win, i_win = acquire_dwell_window(
                    driver, v_now, cycle_start_time, T_DWELL)

            # ── 2) Processamento da janela (média ou fitting) ───────────────
            avg_v, avg_i, info = process_window(t_win, v_win, i_win, METHOD)

            # Conversões para densities
            avg_j = avg_i * 1000.0 / SAMPLE_AREA                 # mA/cm²
            avg_p_mW = abs(avg_v * avg_i * 1000.0)               # mW (total)
            avg_p_density = abs(avg_v * avg_j)                   # mW/cm²
            avg_pce = avg_p_mW / (P_IN * SAMPLE_AREA) * 100.0    # %

            # Janela vazia / sem leitura válida -> pula o ciclo
            if not np.isfinite(avg_i) or not np.isfinite(avg_v):
                print("  (janela vazia — ciclo ignorado)")
                continue

            # Registra amostras brutas desta janela (tempo ABSOLUTO desde o início)
            for tt, vv, ii in zip(t_win, v_win, i_win):
                jj = ii * 1000.0 / SAMPLE_AREA
                pp = abs(vv * ii * 1000.0)
                pd_ = abs(vv * jj)
                raw_records.append({
                    "time(s)": float(tt) + (cycle_t0 - cycle_start_time),
                    "cycle": cycle,
                    "voltage(V)": float(vv),
                    "current(A)": float(ii),
                    "j_current(mA/cm²)": float(jj),
                    "power(mW)": float(pp),
                    "power_density(mW/cm²)": float(pd_),
                    "method": METHOD,
                })

            # ── 3) Lógica Perturb & Observe ─────────────────────────────────
            v_now, direction, is_exploring, step = po_logic(
                avg_p_mW, avg_v, direction, is_exploring, p_prev)
            p_prev = avg_p_mW

            # Clamp de segurança: mantém a tensão dentro do range do instrumento
            v_max = getattr(driver, "V_MAX_V", None) or 2.0
            if v_now < 0.0 or v_now > v_max:
                v_now = min(max(v_now, 0.0), v_max)
                print(f"  (tensão limitada ao range [0, {v_max}] V)")

            mode = "REFINE" if not is_exploring else "EXPLORE"

            # Atualiza o MPP global, se necessário
            if avg_p_mW > mpp["p_mW"]:
                mpp = {"v": avg_v, "j": avg_j, "p_mW": avg_p_mW,
                       "pce": avg_pce, "cycle": cycle}

            po_records.append({
                "cycle": cycle,
                "time(s)": float(t_win[-1]) + (cycle_t0 - cycle_start_time)
                           if len(t_win) else np.nan,
                "voltage(V)": float(avg_v),
                "j_current(mA/cm²)": float(avg_j),
                "power(mW)": float(avg_p_mW),
                "power_density(mW/cm²)": float(avg_p_density),
                "PCE(%)": float(avg_pce),
                "direction": int(direction),
                "step(V)": float(step),
                "mode": mode,
            })

            print(f"Cycle {cycle:3d} | t={t_win[-1] if len(t_win) else 0:7.2f}s | "
                  f"V={avg_v:6.3f} V | J={avg_j:7.3f} mA/cm² | "
                  f"P={avg_p_mW:7.3f} mW | PCE={avg_pce:6.3f}% | {mode}")

            cycle += 1

            # Garante um tempo mínimo de ciclo (evita loop vazio em hardware rápido)
            cycle_dt = time.perf_counter() - cycle_t0
            if cycle_dt < MIN_CYCLE_TIME:
                time.sleep(MIN_CYCLE_TIME - cycle_dt)

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    except Exception as e:
        print(f"Erro durante o MPPT: {e}")
    finally:
        try:
            driver.disconnect()
        except Exception:
            pass
        print("MPPT finished!")

    # ── 4) Saída (plot + CSV) ───────────────────────────────────────────────
    if not raw_records:
        print("Nenhum dado adquirido; abortando geração de saída.")
        return

    raw_df = pd.DataFrame(raw_records)
    po_df = pd.DataFrame(po_records)
    plot_and_save(raw_df, po_df, mpp, smu_dir, INSTRUMENT, METHOD)


if __name__ == "__main__":
    main()
