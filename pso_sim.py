import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


class MPPT_PSO:
    """ Sequential MPPT PSO algorithm for solar PV systems under partial shading.

    Complete iteration consumes n_particles hardware cycles.
    Each step call evaluates the current particle and advances the particle index.
    After completing all particles in a generation, velocities and positions
    of all particles are updated simultaneously, advancing to the next generation.
    """
    def __init__(
            self,
            v_min: float = 0.0,
            v_max: float = 1.1,
            n_particles: int = 5,
            w_max: float = 0.9,                        # max inertia value
            w_min: float = 0.4,                        # min inertia value
            c1: float = 2.0,                           # cognitive acceleration constant
            c2: float = 2.0,                           # social acceleration constant
            v_step_max_frac: float = 0.15,             # (v_max - v_min) fraction
            max_iterations: int = 20,                  # max iterations before force convergence
            stagnation_tol: float = 1e-3,              # mW, min power improvement threshold
            stagnation_patience: int = 3,              # iterations without improvement -> converge
            reacquire_drop_frac: float = 0.10,         # P drop fraction to exit HOLD mode
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

        # Uniform initialization across [v_min, v_max]
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
        self.mode = "PSO"               # "PSO" - EXPLORE; "HOLD" - CONVERGED / MONITORING

    def step(self, avg_p: float) -> float:
        """ Evaluates current particle power (fitness) and returns next voltage setpoint. """
        if self.mode == "HOLD":
            return self._monitor(avg_p)

        i = self.particle_idx
        fitness = float(avg_p)

        # Update Personal Best
        if fitness > self.pbest_fit[i]:
            self.pbest_fit[i] = fitness
            self.pbest_pos[i] = self.positions[i]

        # Update Global Best
        if fitness > self.gbest_fit:
            self.gbest_fit = fitness
            self.gbest_pos = float(self.positions[i])

        self.particle_idx += 1

        # End of generation -> update swarm velocities and positions
        if self.particle_idx >= self.n:
            self._update_swarm()
            self.particle_idx = 0
            self.iteration += 1
            self._check_convergence()

        if self.converged:
            return self.gbest_pos

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

        # Boundary reflection: bounce back inside [v_min, v_max] and negate velocity
        below = self.positions < self.v_min
        above = self.positions > self.v_max
        self.positions[below] = self.v_min
        self.positions[above] = self.v_max
        self.velocities[below] *= -0.5
        self.velocities[above] *= -0.5

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
                f"INFO: PSO converged -- V_mpp~{self.gbest_pos:.4f} V, "
                f"P~{self.gbest_fit:.3f} mW, after {self.iteration} generations "
                f"({self.n * self.iteration} hardware cycles)."
            )

    def _monitor(self, avg_p: float) -> float:
        drop_frac = 0.0
        if self.gbest_fit > 0:
            drop_frac = (self.gbest_fit - avg_p) / self.gbest_fit
        if drop_frac > self.reacquire_drop_frac:
            print(f"INFO: Power drop of {drop_frac*100:.1f}% detected. Restarting PSO...")
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
            return float(self.positions[0])

        if self.gbest_fit < avg_p:
            self.gbest_fit = float(avg_p)

        return self.gbest_pos


class MPPT_PO:
    """ Standard Perturb and Observe (P&O) MPPT algorithm. """
    def __init__(self, v_start: float = 0.20, v_step: float = 0.015, v_min: float = 0.0, v_max: float = 1.1):
        self.v_now = v_start
        self.v_step = v_step
        self.v_min = v_min
        self.v_max = v_max
        self.direction = 1
        self.p_prev = 0.0
        self.initialized = False

    def step(self, avg_p: float) -> float:
        if not self.initialized:
            self.p_prev = avg_p
            self.initialized = True
        else:
            dP = avg_p - self.p_prev
            if dP < 0:
                self.direction *= -1
            self.p_prev = avg_p

        self.v_now = np.clip(self.v_now + self.v_step * self.direction, self.v_min, self.v_max)
        return float(self.v_now)


def pv_sim_curve(v):
    """ Multi-modal PV curve -- partial shading with several local maxima.

    Uses a Rastrigin-like landscape scaled to the [0, 1.1] V domain so
    there are shallow local wells and a broad global peak.
    """
    A = 2.0
    p_osc = A * (v / 1.1) ** 2 - A * np.cos(2.0 * np.pi * 10.0 * v / 1.1)
    p_peak = 35.0 * np.exp(-0.5 * ((v - 0.72) / 0.14) ** 2)
    return p_osc + p_peak


def main():
    v_min, v_max = 0.0, 1.1
    v = np.linspace(v_min, v_max, 3000)
    p = pv_sim_curve(v)

    # Find theoretical global optimum for reference
    global_idx = np.argmax(p)
    v_global = v[global_idx]
    p_global = p[global_idx]

    # Algorithm parameters
    n_particles = 5
    c1 = 2.0
    c2 = 1.5
    w_max = 0.70
    w_min = 0.20
    v_step_max_frac = 0.20
    max_iterations = 20
    stagnation_patience = 4
    total_hardware_cycles = 100

    # -------------------------------------------------------------------------
    # 1) SIMULATE PSO MPPT
    # -------------------------------------------------------------------------
    pso = MPPT_PSO(
        v_min=v_min, v_max=v_max,
        n_particles=n_particles,
        max_iterations=max_iterations,
        stagnation_patience=stagnation_patience,
        v_step_max_frac=v_step_max_frac,
        w_max=w_max, w_min=w_min,
        c1=c1, c2=c2,
        rng_seed=42,
    )

    gen_snapshots = [pso.positions.copy()]
    pso_v_inst = []
    pso_p_inst = []
    pso_p_gbest = []

    v_setpoint_pso = float(pso.positions[0])

    for cycle in range(total_hardware_cycles):
        p_meas = float(pv_sim_curve(v_setpoint_pso))
        pso_v_inst.append(v_setpoint_pso)
        pso_p_inst.append(p_meas)

        old_iter = pso.iteration
        v_setpoint_pso = pso.step(p_meas)
        pso_p_gbest.append(pso.gbest_fit)

        if pso.iteration != old_iter and pso.mode == "PSO":
            gen_snapshots.append(pso.positions.copy())

    # -------------------------------------------------------------------------
    # 2) SIMULATE PERTURB & OBSERVE (P&O) MPPT
    # -------------------------------------------------------------------------
    po = MPPT_PO(v_start=0.20, v_step=0.015, v_min=v_min, v_max=v_max)
    po_v_inst = []
    po_p_inst = []
    v_setpoint_po = po.v_now

    for cycle in range(total_hardware_cycles):
        p_meas = float(pv_sim_curve(v_setpoint_po))
        po_v_inst.append(v_setpoint_po)
        po_p_inst.append(p_meas)
        v_setpoint_po = po.step(p_meas)

    # Calculate efficiencies
    eff_pso = (pso_p_gbest[-1] / p_global) * 100.0
    eff_po = (po_p_inst[-1] / p_global) * 100.0

    print("==========================================================")
    print("               MPPT SIMULATION COMPARISON")
    print("==========================================================")
    print(f"Global Maximum (Theoretical) : V = {v_global:.4f} V | P = {p_global:.3f} mW")
    print(f"PSO Found                    : V = {pso.gbest_pos:.4f} V | P = {pso_p_gbest[-1]:.3f} mW (Efficiency: {eff_pso:.2f}%)")
    print(f"P&O Found                    : V = {po_v_inst[-1]:.4f} V | P = {po_p_inst[-1]:.3f} mW (Efficiency: {eff_po:.2f}%)")
    print("==========================================================")

    # -------------------------------------------------------------------------
    # 3) ANIMATION -- Particle Evolution over the PV Curve
    # -------------------------------------------------------------------------
    n_frames = len(gen_snapshots)
    fig_anim, ax_anim = plt.subplots(figsize=(9, 5))
    param_text = f"$N={n_particles}$, $c_1={c1}$, $c_2={c2}$, $w=[{w_min}, {w_max}]$, $v_{{step,max}}={v_step_max_frac}\\times\\Delta V$"

    ax_anim.plot(v, p, lw=1.5, color="#3a86ff", zorder=1, label="Fitness curve P(V)")
    ax_anim.axvline(v_global, color="gray", ls="--", lw=0.8, alpha=0.6, label="Global optimum V_MPP")
    ax_anim.axhline(p_global, color="green", ls=":", lw=0.8, alpha=0.5, label=f"Global max = {p_global:.3f} mW")
    ax_anim.set_xlabel(f"Applied Voltage V (V)\n{param_text}")
    ax_anim.set_ylabel("Power P (mW)")
    ax_anim.grid(True, alpha=0.3)

    particles_scatter = ax_anim.scatter([], [], s=60, c="#ff006e", edgecolors="k", linewidths=0.6, zorder=3, label="Particles")
    gbest_marker = ax_anim.scatter([], [], s=140, marker="*", c="#ffbe0b", edgecolors="k", linewidths=0.7, zorder=4, label="gbest")
    ax_anim.legend(loc="upper left", fontsize=8)

    trail_artists = []

    def init():
        particles_scatter.set_offsets(np.empty((0, 2)))
        gbest_marker.set_offsets(np.empty((0, 2)))
        return [particles_scatter, gbest_marker]

    def update(frame):
        nonlocal trail_artists
        positions = gen_snapshots[frame]
        powers = pv_sim_curve(positions)

        for art in trail_artists:
            art.remove()
        trail_artists.clear()

        lookback = min(frame, 5)
        for k in range(1, lookback + 1):
            past = gen_snapshots[frame - k]
            alpha = 0.15 * (1 - k / (lookback + 1))
            tr = ax_anim.scatter(past, pv_sim_curve(past), s=20, c="#ff006e", alpha=alpha, zorder=2)
            trail_artists.append(tr)

        particles_scatter.set_offsets(np.column_stack([positions, powers]))
        best_idx = np.argmax(powers)
        gbest_v = positions[best_idx]
        gbest_p = powers[best_idx]
        gbest_marker.set_offsets([[gbest_v, gbest_p]])

        ax_anim.set_title(f"PSO Particle Evolution -- Generation {frame}/{n_frames - 1}", fontsize=10, fontweight="bold")
        return [particles_scatter, gbest_marker] + trail_artists

    anim = FuncAnimation(fig_anim, update, frames=n_frames, init_func=init, interval=1200, blit=False, repeat=True)
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # 4) STATIC PLOT -- Power & Voltage Evolution Comparison (PSO vs P&O)
    # -------------------------------------------------------------------------
    fig_conv, (ax_p, ax_v_plot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    cycles = np.arange(total_hardware_cycles)

    # --- Top Panel: Power Evolution ---
    ax_p.plot(cycles, pso_p_inst, "o--", color="#ff7f0e", markersize=3, alpha=0.4, label="PSO Instantaneous P")
    ax_p.plot(cycles, pso_p_gbest, "-", color="#1f77b4", lw=2.2, label=f"PSO gbest (Final: {pso_p_gbest[-1]:.2f} mW)")
    ax_p.plot(cycles, po_p_inst, "-", color="#d62728", lw=1.8, label=f"P&O Tracked P (Final: {po_p_inst[-1]:.2f} mW)")
    ax_p.axhline(p_global, color="#2ca02c", ls="--", lw=1.5, label=f"Global Max ({p_global:.2f} mW)")

    ax_p.set_ylabel("Power P (mW)", fontsize=11, fontweight="bold")
    ax_p.set_title("MPPT Power Evolution: PSO vs Perturb & Observe (P&O)", fontsize=12, fontweight="bold")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white")

    # Annotations
    ax_p.annotate(f"P&O Trapped at Local Peak\n({po_p_inst[-1]:.2f} mW, eff={eff_po:.1f}%)",
                  xy=(total_hardware_cycles - 15, po_p_inst[-1]),
                  xytext=(total_hardware_cycles - 35, po_p_inst[-1] - 8),
                  arrowprops=dict(facecolor="#d62728", shrink=0.05, width=1, headwidth=6),
                  fontsize=9, fontweight="bold", color="#d62728")

    ax_p.annotate(f"PSO Global MPP Found\n({pso_p_gbest[-1]:.2f} mW, eff={eff_pso:.1f}%)",
                  xy=(25, pso_p_gbest[25]),
                  xytext=(30, pso_p_gbest[25] - 10),
                  arrowprops=dict(facecolor="#1f77b4", shrink=0.05, width=1, headwidth=6),
                  fontsize=9, fontweight="bold", color="#1f77b4")

    # --- Bottom Panel: Voltage Evolution ---
    ax_v_plot.plot(cycles, pso_v_inst, "o--", color="#ff7f0e", markersize=3, alpha=0.4, label="PSO Instantaneous V")
    ax_v_plot.plot(cycles, [pso.gbest_pos]*total_hardware_cycles, "-", color="#1f77b4", lw=2.0, label="PSO gbest V")
    ax_v_plot.plot(cycles, po_v_inst, "-", color="#d62728", lw=1.8, label="P&O Operating V")
    ax_v_plot.axhline(v_global, color="#2ca02c", ls="--", lw=1.5, label=f"Global V_MPP ({v_global:.3f} V)")

    ax_v_plot.set_xlabel("Hardware Cycle", fontsize=11, fontweight="bold")
    ax_v_plot.set_ylabel("Applied Voltage V (V)", fontsize=11, fontweight="bold")
    ax_v_plot.set_title("Applied Voltage Setpoint Dynamics", fontsize=11, fontweight="bold")
    ax_v_plot.grid(True, alpha=0.3)
    ax_v_plot.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white")

    plt.tight_layout()

    output_dir = Path("output MPPT")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "pso_vs_po_comparison.png"
    fig_conv.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {save_path}")

    plt.show()
    return anim, fig_conv


if __name__ == "__main__":
    _anim, _fig = main()