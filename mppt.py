import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from jv_sweep import build_driver, preconditioning_device
import time

try:
    from scipy.optimize import curve_fit
except ImportError:  # scipy optional: only needed for the "fitting" method
    curve_fit = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION  –  INSTRUMENT, METHOD AND MPPT PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUMENT = "KEITHLEY"   # "USMU" | "KEITHLEY" | "ADALM1000" | "AD3"
OUTPUT_DIR  = Path("output MPPT")
OUTPUT_DIR.mkdir(exist_ok=True)

# Acquisition method / steady-state determination:
#   "cv"      -> sample until the current coefficient of variation (CV) drops
#                below MIN_CV (stability criterion)
#   "fixed"   -> wait a fixed dwell time (T_DWELL) and use the average
#   "fitting" -> wait T_DWELL and extrapolate the steady-state current
#                via double-exponential transient fit
METHOD = "cv"               # "cv" / "fitting" / "fixed"
CV_WINDOW = 20              # number of samples in the sliding CV window
MIN_CV = 0.1              # maximum CV (0.1%) to consider steady state

# CORE MPPT LOGIC
LOGIC = "PO"        # "PO" - Perturb and Observe, "INC" - Incremental Conductance or "PSO"

# Initial step direction for Perturb & Observe algorithm:
#   "FORWARD"    -> increase voltage (direction = +1)
#   "REVERSE"    -> decrease voltage  (direction = -1)
#   "FROM_VOC"   -> precondition, start from Voc and decrease (direction = -1)
ORIENTATION = "FORWARD"     # "FORWARD" / "REVERSE" / "FROM_VOC"

# Control approach.
APPROACH = "POTENTIOSTATIC"   # "POTENTIOSTATIC" / "GALVANOSTATIC"

# INC method
epsilon = 1e-4   # dead band for INC logic
It = 1e-3     # current limit to leave MPP (mA)

# ── MPPT Parameters (Perturb & Observe) ───────────────────────────────────
V_START    = 0.6    # V  – initial tracking voltage
I_START    =-29.1e-3    # A  - initial galvanostatic MPPT current (±10mA para ADALM1000)
T_DWELL    = 2       # s  – tempo de dwell (métodos "fixed" / "fitting")
LARGE_STEP = 0.05     # V  – perturbation step during EXPLORATION phase
SMALL_STEP = 0.01    # V  – perturbation step during REFINEMENT phase (near MPP)
LARGE_I_STEP = 1e-3  # I – perturbation step during EXPLORATION phase no modo galvanostático
SMALL_I_STEP = 0.5e-3# I – perturbation step during EXPLORATION phase no modo galvanostático
SAMPLE_AREA = 0.16      # cm²
P_IN       = 100     # mW/cm²  – irradiância incidente (para o cálculo de PCE)
T_TOTAL    = 100      # s  – duração total do rastreamamento
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
def acquire_cv_window(driver, excitation_now, cycle_start_time):
    """Aplica V=excitation_now (ou I=excitation_now) e amostra até a corrente/tensão estabilizar (CV < MIN_CV) ou TIMEOUT.

    A janela é cronometrada a partir de seu próprio início (window_start),
    não do início do tracking — o TIMEOUT e o t_rel referem-se ao ciclo atual.

    Retorna (t, v, i) como arrays numpy (tempo relativo ao início da janela).
    """
    t_buf, v_buf, i_buf = [], [], []
    steady_state = False
    cv_buf = []
    window_start = time.perf_counter()
    while (not steady_state) and (time.perf_counter() - window_start) < TIMEOUT:
        if APPROACH == "POTENTIOSTATIC":
            vv, ii = driver.set_voltage_and_measure(excitation_now)
        else:
            vv, ii = driver.set_current_and_measure(excitation_now)
        t_rel = time.perf_counter() - window_start
        t_buf.append(t_rel)
        v_buf.append(vv)
        i_buf.append(ii)

        # Monitora a corrente no modo potenciostático ou a tensão no modo galvanostático
        signal_val = ii if APPROACH == "POTENTIOSTATIC" else vv
        cv_buf.append(signal_val)
        if len(cv_buf) > CV_WINDOW:
            del cv_buf[0]
            sig_mean = np.mean(cv_buf)
            sig_std = np.std(cv_buf)
            if sig_mean != 0 and abs(sig_std / sig_mean) < MIN_CV:
                steady_state = True

    if not steady_state:
        print("[cv] TIMEOUT atingido sem estabilidade — usando janela completa.")
    return np.array(t_buf), np.array(v_buf), np.array(i_buf)


def acquire_dwell_window(driver, excitation_now, cycle_start_time, dwell):
    """Aplica V=v_now e amostra por 'dwell' segundos (métodos fixed/fitting).

    A janela é cronometrada a partir de seu próprio início (window_start),
    não do início do tracking — caso contrário ciclos posteriores adquirem
    janelas de duração ~0 (bug: o while usava cycle_start_time como referência).

    Retorna (t, v, i) como arrays numpy (tempo relativo ao início da janela).
    """
    t_buf, v_buf, i_buf = [], [], []
    window_start = time.perf_counter()
    while (time.perf_counter() - window_start) < dwell:
        if APPROACH == "POTENTIOSTATIC":
            v_meas, i_meas = driver.set_voltage_and_measure(excitation_now)
        else:
            # Galvanostatic mode
            v_meas, i_meas = driver.set_current_and_measure(excitation_now)
        t_rel = time.perf_counter() - window_start
        t_buf.append(t_rel)
        v_buf.append(v_meas)
        i_buf.append(i_meas)
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
# PERTURB & OBSERVE LOGIC (PO)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def po_logic(p_now, excitation_now, direction, is_exploring, p_prev):
    """Decide o próximo passo do rastreamento MPPT.

    Parâmetros
    ----------
    p_now         : potência média desta janela (mW, valor absoluto)
    excitation_now: setpoint atual de tensão (V) ou corrente (A)
    direction     : direção atual do passo (+1 sobe setpoint, -1 desce setpoint)
    is_exploring  : True se ainda procura a região do MPP (passo grande)
    p_prev        : potência da janela anterior (mW, abs); 0 no 1º ciclo

    Retorna
    -------
    (excitation_next, direction, is_exploring, step)
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

    if APPROACH == "POTENTIOSTATIC":
        step = LARGE_STEP if is_exploring else SMALL_STEP
        v_next = excitation_now + step * direction
        return v_next, direction, is_exploring, step
    else:
        step = LARGE_I_STEP if is_exploring else SMALL_I_STEP
        i_next = excitation_now + step * direction
        return i_next, direction, is_exploring, step

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INCREMENTAL CONDUCTANCE LOGIC (INC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def inc_logic(avg_i, avg_v, i_prev, v_prev, excitation_now, direction=1, is_exploring=True):
    """Decide the next MPPT tracking step via Incremental Conductance (INC) method.

    No MPP (ou quando dV == 0), mantém o potencial constante (direction = 0)
    até que a variação de corrente |dI| ultrapasse o limite 'It'.

    Parâmetros
    ----------
    avg_i         : corrente média medida nesta janela (A)
    avg_v         : tensão média medida nesta janela (V)
    i_prev        : corrente da janela anterior (A)
    v_prev        : tensão da janela anterior (V)
    excitation_now: setpoint atual de tensão (V) ou corrente (A)
    direction     : direção atual do passo (+1, -1 ou 0)
    is_exploring  : True se na fase de busca (passo grande), False no refino (passo menor)

    Retorna
    -------
    (excitation_next, direction, is_exploring, step)
    """
    dI = abs(avg_i) - abs(i_prev)
    dV = avg_v - v_prev

    # 1) Se dV == 0 (está segurando no mesmo potencial Vmpp / Impp)
    if abs(dV) < 1e-6:
        if dI != 0:
            if abs(dI) > It:
                # Variação na iluminação/corrente acima do limite: retoma o rastreamento
                direction = 1 if dI > 0 else -1
                print(f"INFO: Variação de corrente detectada (|dI|={abs(dI)*1000:.3f} mA > {It*1000:.3f} mA) — retomando rastreamento.")
            else:
                # Variação dentro do limite: mantém segurado em Vmpp (sem oscilação)
                direction = 0
    else:
        # 2) Se dV != 0, avalia a condutância incremental dI/dV + I/V
        if avg_v != 0:
            inc_val = (dI / dV) + (abs(avg_i) / avg_v)
            if abs(inc_val) < epsilon:
                # Chegou ao MPP: trava no potencial atual (direction = 0)
                direction = 0
                if is_exploring:
                    is_exploring = False
                print(f"INFO: MPP encontrado (INC ~ 0, epsilon real = {abs(inc_val)}) em V = {avg_v:.4f} V, I = {avg_i*1000:.3f} mA — segurando potencial.")
            elif inc_val > 0:
                # À esquerda do MPP (V < V_mpp) -> aumentar V (ou aumentar I em direção a 0 A)
                if direction != 1 and is_exploring:
                    is_exploring = False
                    print("INFO: Direção invertida — região do MPP encontrada (REFINO).")
                direction = 1
            else:
                # À direita do MPP (V > V_mpp) -> diminuir V (ou diminuir I em direção a -I_sc)
                if direction != -1 and is_exploring:
                    is_exploring = False
                    print("INFO: Direção invertida — região do MPP encontrada (REFINO).")
                direction = -1

    if APPROACH == "POTENTIOSTATIC":
        step = LARGE_STEP if is_exploring else SMALL_STEP
        v_next = excitation_now + step * direction
        return v_next, direction, is_exploring, step
    else:
        step = LARGE_I_STEP if is_exploring else SMALL_I_STEP
        i_next = excitation_now + step * direction
        return i_next, direction, is_exploring, step

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARTICLE SWARM OPTIMIZATION (PSO)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MPPT_PSO:
    """ Sequential MPPT PSO algorithm

    Complete iteration consumes n_particles
    Hardware cycles: Each step call evaluate the current particle
    and increase index
    After completing the swarm, velocities and positions
    of all particles are updated at the same time and a new generation begins

    """
    def __init__(
            self,
            v_min: float = 0.0,
            v_max: float = 1.1,
            n_particles: int = 5,
            w_max = 0.9,                        # max inertia value
            w_min = 0.4,                        # min
            c1: float = 2.0,                    # cognitive acceleration constant
            c2: float = 2.0,                    # social acceleration constant
            v_step_max_frac: float =  0.15,     # (v_max - v_min) fraction
            max_iterations = 20,               # n iterations before force convergence
            stagnation_tol: float = 1e-3,       # mW, min power improve
            stagnation_patience: int = 3,       # iterations without improvement -> converge
            reacquire_drop_frac: float = 0.1,    # P drop to exit HOLD mode
            rng_seed: int | None = None
):
        self.v_min = v_min
        self.v_max = v_max
        self.n = n_particles
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_step_max = v_step_max_frac * (v_max - v_min)
        self.max_iterations = max_iterations
        self.stagnation_tol = stagnation_tol
        self.stagnation_patience = stagnation_patience
        self.reacquire_drop_frac = reacquire_drop_frac
        self._v_step_max_frac = v_step_max_frac

        self._rng = np.random.default_rng(rng_seed)

        # Initialization: Uniform distribution of the particles
        base = np.linspace(v_min, v_max, n_particles, endpoint=False)
        slice_width = (v_max - v_min) / n_particles
        jitter = self._rng.uniform(0.0, slice_width, size=n_particles)
        self.positions = base + jitter
        self.velocities = self._rng.uniform(-self.v_step_max, self.v_step_max, size=n_particles)

        self.pbest_pos = self.positions.copy()
        self.pbest_fit = np.full(n_particles, -np.inf)

        self.gbest_pos = float(self.positions[0])
        self.gbest_fit = -np.inf
        self._last_gbest_fit = -np.inf

        self.particle_idx = 0
        self.iteration = 0
        self.stagn_count = 0
        self.converged = False
        self.mode = "PSO"               # "PSO" - EXPLORE; "HOLD" - CONVERGED; MONITORING

    def step(self, avg_p: float) -> float:
        if self.mode == "HOLD":
            return self._monitor(avg_p)

        i = self.particle_idx
        fitness = float(avg_p)

        if fitness > self.pbest_fit[i]:
            self.pbest_fit[i] = fitness
            self.pbest_pos[i] = self.positions[i]

        if fitness > self.gbest_fit:
            self.gbest_fit = fitness
            self.gbest_pos = float(self.positions[i])

        self.particle_idx += 1

        if self.particle_idx >= self.n:
            self._update_swarm()
            self.particle_idx = 0
            self.iteration += 1
            self._check_convergence()

        # If convergence was just detected, hold at the best position
        # instead of jumping to a random particle after swarm update.
        if self.converged:
            return self.gbest_pos

        # Return the position of the next particle to evaluate
        return float(self.positions[self.particle_idx])

    def _update_swarm(self):
        frac = self.iteration / max(self.max_iterations, 1)
        w = self.w_max - (self.w_max - self.w_min) * frac
        w = max(w, self.w_min)

        r1 = self._rng.uniform(0.0, 1.0, size=self.n)
        r2 = self._rng.uniform(0.0, 1.0, size=self.n)

        cognitive = self.c1 * r1 * (self.pbest_pos - self.positions)
        social = self.c2 * r2 * (self.gbest_pos - self.positions)
        self.velocities = w * self.velocities + cognitive + social

        # Constrain velocity
        self.velocities = np.clip(self.velocities, -self.v_step_max, self.v_step_max)

        self.positions += self.velocities

        # Edge reflection
        below = self.positions < self.v_min
        above = self.positions > self.v_max
        self.positions[below] = self.v_min
        self.positions[above] = self.v_max
        self.velocities[below] *= 0.5
        self.velocities[above] *= 0.5

    def _check_convergence(self):
        improvement = self.gbest_fit - self._last_gbest_fit
        self._last_gbest_fit = self.gbest_fit

        if improvement < self.stagnation_tol:
            self.stagn_count += 1
        else:
            self.stagn_count = 0

        if self.stagn_count >= self.stagnation_patience or self.iteration >= self.max_iterations:
            self.converged = True
            self.mode = "HOLD"
            print(
                f"INFO: PSO converged — V_mpp≈{self.gbest_pos:.4f} V, "
                f"P≈{self.gbest_fit:.3f} mW, after {self.iteration} generations "
                f"({self.n * self.iteration} hardware cycles)."
            )

    def _monitor(self, avg_p: float) -> float:
        drop_frac = 0
        if self.gbest_fit > 0:
            drop_frac = (self.gbest_fit - avg_p) / self.gbest_fit
        if drop_frac > self.reacquire_drop_frac:
            print(f"INFO: Power drop of {drop_frac*100:.1f} % detected"
                  f". Restarting PSO...")

            self.__init__(
                v_min=self.v_min,
                v_max=self.v_max,
                n_particles=self.n,
                w_max=self.w_max,
                w_min=self.w_min,
                c1=self.c1,
                c2=self.c2,
                v_step_max_frac=self._v_step_max_frac,
                max_iterations=self.max_iterations,
                stagnation_tol=self.stagnation_tol,
                stagnation_patience=self.stagnation_patience,
                reacquire_drop_frac=self.reacquire_drop_frac,
            )
            # Return the first particle position WITHOUT evaluating it.
            # The next cycle will measure the actual power at this position
            # and call step() with the correct fitness.
            return float(self.positions[0])

        if self.gbest_fit < avg_p:
            self.gbest_fit = float(avg_p)

        return self.gbest_pos


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT E SAVE (salva os dados brutos + resumo do tracking + figuras)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_and_save(raw_df, po_df, mpp, smu_dir, instrument, method):
    """Generate CSVs and scientific plots for MPPT tracking.

    Figuras:
      1 - tensão bruta vs tempo
      2 - corrente bruta vs tempo
      3 - PCE (ou densidade de potência) por ciclo  -> convergence to MPP
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
    if LOGIC == "PO":
        ax.set_xlabel("PO Cycle")
    elif LOGIC == "INC":
        ax.set_xlabel("INC Cycle")
    elif LOGIC == "PSO":
        ax.set_xlabel("PSO Cycle")

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
    print(f"\nConnecting to instrument: {INSTRUMENT}")
    driver = build_driver(INSTRUMENT)
    smu_dir = OUTPUT_DIR / INSTRUMENT / "kasia_3_1108" / APPROACH / LOGIC / METHOD / ORIENTATION
    smu_dir.mkdir(parents=True, exist_ok=True)

    if APPROACH == "POTENTIOSTATIC":
        # Potentiostatic
        driver.connect(curr_sour=False)
    else:
        # Galvanostatic
        driver.connect(curr_sour=True)

    v_start = V_START
    i_start = I_START

    # FROM_VOC: pré-condiciona e parte do Voc medido
    if ORIENTATION == "FROM_VOC":
        if hasattr(driver, "set_source_mode"):
            driver.set_source_mode(curr_sour=True)
        try:
            voc = preconditioning_device(driver)
            v_start = voc
            i_start = 0.0  # Em Voc a corrente do gerador é ~0 A
            print(f"FROM_VOC: starting tracking from Voc = {v_start:.4f} V (I = 0.0 A)")
        except Exception as e:
            v_start = V_START
            i_start = I_START
            print(f"FROM_VOC preconditioning falhou ({e}); usando V_START={v_start} V, I_START={i_start} A")
        if hasattr(driver, "set_source_mode"):
            driver.set_source_mode(curr_sour=False if APPROACH == "POTENTIOSTATIC" else True)

    # Configura integração / NPLC (após voltar ao modo tensão)
    if INSTRUMENT == "USMU":
        driver.configure_integration(1)
    elif INSTRUMENT == "KEITHLEY":
        driver.configure_integration(1)

    # Direção inicial conforme a orientação escolhida
    direction = 1 if ORIENTATION == "FORWARD" else -1

    # Estado do algoritmo
    is_exploring = True       # True=EXPLORAÇÃO (passo grande), False=REFINO
    if APPROACH == "POTENTIOSTATIC":
        excitation_now = v_start
    else:
        # Galvanostático: excitation_now = corrente no setpoint
        excitation_now = i_start
    p_prev = 0.0              # previous window power (abs, mW)

    # Best maximum power point (MPP) found so far
    mpp = {"v": np.nan, "j": np.nan, "p_mW": -np.inf, "pce": np.nan, "cycle": -1}

    # Acumuladores
    raw_records = []          # uma linha por amostra bruta
    cycle_records = []           # one row per PO or INC cycle

    # Initialize PSO state
    pso = MPPT_PSO(
        v_min=0.6, v_max=1.0,
        n_particles=4,
        max_iterations=15,
        stagnation_patience=3,
        v_step_max_frac=0.25,
        w_max=0.9, w_min=0.4,
        c1=2.0, c2=2.0,
        rng_seed=42,
    )


    print("Starting MPPT tracking...")
    cycle_start_time = time.perf_counter()
    cycle = 0
    try:
        while (time.perf_counter() - cycle_start_time) < T_TOTAL:
            cycle_t0 = time.perf_counter()
            # ── 1) Aquisição da janela ──────────────────────────────────────
            if METHOD == "cv":
                t_win, v_win, i_win = acquire_cv_window(driver, excitation_now, cycle_start_time)
            else:  # "fixed" ou "fitting"
                t_win, v_win, i_win = acquire_dwell_window(
                    driver, excitation_now, cycle_start_time, T_DWELL)

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

            # ── 3) Lógica do Algoritmo ─────────────────────────────────
            if LOGIC == "PO":
                excitation_now, direction, is_exploring, step = po_logic(
                    avg_p_mW, excitation_now, direction, is_exploring, p_prev)
                p_prev = avg_p_mW
            elif LOGIC == "INC":
                if cycle > 0:
                    i_prev = (cycle_records[-1]["j_current(mA/cm²)"]) * SAMPLE_AREA/1000
                    v_prev = cycle_records[-1]["voltage(V)"]
                else:
                    i_prev = avg_i
                    v_prev = avg_v
                excitation_now, direction, is_exploring, step = inc_logic(
                    avg_i, avg_v, i_prev, v_prev, excitation_now, direction, is_exploring)
            elif LOGIC == "PSO":
                excitation_now = pso.step(avg_p_mW)
                direction = 0
                step = 0.0
                is_exploring = not pso.converged

            # Clamp de segurança: mantém a tensão/corrente dentro do range do instrumento
            if APPROACH == "POTENTIOSTATIC":
                v_max = getattr(driver, "V_MAX_V", None) or 7.0
                if excitation_now < 0.0 or excitation_now > v_max:
                    excitation_now = min(max(excitation_now, 0.0), v_max)
                    print(f"  (tensão limitada ao range [0, {v_max}] V)")
            else:
                # Galvanostático: limite de corrente (não tensão)
                # Correntes negativas = geração (gerador solar), positivas = absorção
                i_max = getattr(driver, "CURRENT_LIMIT_A", None) or 0.1
                if abs(excitation_now) > i_max:
                    excitation_now = min(max(excitation_now, -i_max), i_max)
                    print(f"  (corrente limitada ao range [{-i_max}, {i_max}] A)")

            mode = "HOLD" if (LOGIC == "PSO" and pso.converged) else ("REFINE" if not is_exploring else "EXPLORE")

            # Atualiza o MPP global, se necessário
            if avg_p_mW > mpp["p_mW"]:
                mpp = {"v": avg_v, "j": avg_j, "p_mW": avg_p_mW,
                       "pce": avg_pce, "cycle": cycle}

            cycle_records.append({
                "cycle": cycle,
                "time(s)": float(t_win[-1]) + (cycle_t0 - cycle_start_time)
                           if len(t_win) else np.nan,
                "voltage(V)": float(avg_v),
                "j_current(mA/cm²)": float(avg_j),
                "power(mW)": float(avg_p_mW),
                "power_density(mW/cm²)": float(avg_p_density),
                "PCE(%)": float(avg_pce),
                "direction": int(direction),
                "step": float(step),
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
    po_df = pd.DataFrame(cycle_records)
    plot_and_save(raw_df, po_df, mpp, smu_dir, INSTRUMENT, METHOD)

def analyse_single_transient(file_path=None):
    """Plot and fit negative and positive transients from fixed PO and calculate CV of the last 20 datapoints."""
    if file_path is None:
        file_path = r"C:\Users\hyd-laptop\Documents\Low-cost-SMUs\output MPPT\USMU\yanyan_dev3\POTENTIOSTATIC\PO\fixed\FORWARD\mppt_raw.csv"
    
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Erro: arquivo {file_path} não encontrado.")
        return

    df = pd.read_csv(file_path)
    
    # Identificar ciclos no intervalo 30s a 34s (geralmente ciclos 15 e 16)
    # Ciclo 15: degrau positivo de tensão (V aumenta)
    # Ciclo 16: degrau negativo de tensão (V diminui)
    cycles = [15, 16]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for i_ax, (cyc, ax, title_suffix, color, text_y) in enumerate([
        (15, ax1, "Positive Step", "#1f77b4", 0.50),
        (16, ax2, "Negative Step", "#2ca02c", 0.50)
    ]):
        cyc_df = df[df["cycle"] == cyc]
        if cyc_df.empty:
            print(f"Aviso: ciclo {cyc} não encontrado nos dados.")
            continue
            
        t = cyc_df["time(s)"].values
        i = cyc_df["current(A)"].values
        v = cyc_df["voltage(V)"].values
        
        # Tempo relativo ao início do ciclo/janela
        t_rel = t - t[0]
        
        # Coeficiente de variação (CV) dos últimos 20 pontos
        last_20_i = i[-20:]
        i_mean = np.mean(last_20_i)
        cv = np.std(last_20_i) / np.abs(np.mean(last_20_i)) if len(last_20_i) else np.nan
        
        # Ajuste bi-exponencial
        fit_params = double_exp_fitting(t_rel, i)
        
        # Plot dos dados experimentais
        ax.scatter(t_rel, i * 1000, color=color, label="Experimental Data", s=25, alpha=0.8, edgecolors="none")
        
        if fit_params[0] is not None:
            # Traçar a curva de ajuste
            t_fit = np.linspace(0, t_rel[-1], 200)
            i_fit = double_exp_func(t_fit, *fit_params[:-1])
            ax.plot(t_fit, i_fit * 1000, color="#d62728", label="Bi-Exponential Fit")
            
            # Anotação com os parâmetros do ajuste
            a1, t1, a2, t2, c, r2 = fit_params
            textstr = '\n'.join((
                r'$a_1 = %.3e\ \mathrm{A}$' % a1,
                r'$\tau_1 = %.3f\ \mathrm{ms}$' % (t1 * 1000),
                r'$a_2 = %.3e\ \mathrm{A}$' % a2,
                r'$\tau_2 = %.3f\ \mathrm{s}$' % t2,
                r'$I_{mean} = %.3f\ \mathrm{mA}$' % (i_mean * 1000),
                r'$I_{ss} = %.3f\ \mathrm{mA}$' % (c * 1000),
                r'$R^2 = %.4f$' % r2,
                r'$\mathrm{CV}_{20} = %.4f\%%$' % (cv * 100)
            ))
        else:
            textstr = r'$\mathrm{CV}_{20} = %.4f\%%$' % (cv * 100)
            
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc')
        ax.text(0.65, text_y, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
                
        # Detalhes do gráfico
        v_start = df[df["cycle"] == (cyc - 1)]["voltage(V)"].iloc[-1] if cyc > 0 and not df[df["cycle"] == (cyc - 1)].empty else v[0]
        v_end = v[0]
        ax.set_xlabel("Relative Time (s)")
        ax.set_ylabel("Current (mA)")
        ax.set_title(f"{title_suffix} (Cycle {cyc}: {v_start:.3f}V → {v_end:.3f}V)")
        ax.legend(loc="best")
        
        # Print resumido no console
        print(f"\n--- Análise do Ciclo {cyc} ({title_suffix}) ---")
        print(f"Degrau de tensão: {v_start:.3f} V -> {v_end:.3f} V")
        print(f"Pontos adquiridos: {len(t)}")
        print(f"CV dos últimos 20 pontos: {cv * 100:.6f} %")
        if fit_params[0] is not None:
            print(f"Ajuste Bi-Exponencial:")
            print(f"  a1 = {fit_params[0]:.4e} A, tau1 = {fit_params[1]*1000:.3f} ms")
            print(f"  a2 = {fit_params[2]:.4e} A, tau2 = {fit_params[3]:.3f} s")
            print(f"  Iss (offset c) = {fit_params[4]*1000:.4f} mA")
            print(f"  R2 = {fit_params[5]:.6f}")
        else:
            print("Ajuste falhou ou scipy indisponível.")
            
    fig.suptitle("Análise de Transientes & Ajuste Bi-Exponencial", fontsize=14, weight="bold")
    fig.tight_layout()
    
    # Salvar a imagem no mesmo diretório do arquivo raw
    output_path = file_path.parent / "single_transient_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n[analyse] Gráfico científico salvo em: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
