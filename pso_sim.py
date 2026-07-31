import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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

def pv_sim_curve(v):
    """ Multi-modal PV curve — partial shading with several local maxima.

    Uses a Rastrigin-like landscape scaled to the [0, 1.1] V domain so
    there are ~10 well-spaced peaks.  Returns positive power (PSO
    maximises by convention).
    """
    A = 2.0
    # 10 full cosine periods over [0, 1.1] → 10 shallow wells
    p_osc = A * (v / 1.1) ** 2 - A * np.cos(2.0 * np.pi * 10.0 * v / 1.1)
    # Single broad Gaussian — global peak near MPP region
    p_peak = 35.0 * np.exp(-0.5 * ((v - 0.72) / 0.14) ** 2)
    return p_osc + p_peak

def main():
    v_min, v_max = 0.5, 0.9
    v = np.linspace(v_min, v_max, 3000)
    p = pv_sim_curve(v)

    # Find global optimum for reference
    global_idx = np.argmax(p)
    v_global = v[global_idx]
    p_global = p[global_idx]

    # PSO parameters configuration
    n_particles = 5
    c1 = 2.0
    c2 = 2.0
    w_max = 0.50
    w_min = 0.20
    v_step_max_frac = 0.25
    max_iterations = 20
    stagnation_patience = 5

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

    # Store snapshots: particle positions at the START of each generation
    gen_snapshots = [pso.positions.copy()]       # generation 0 initial spread
    history_fit = []
    history_pos = []

    for _ in range(500):
        if pso.mode == "PSO":
            old_iter = pso.iteration
            fitness = float(pv_sim_curve(pso.positions[pso.particle_idx]))
            pso.step(fitness)
            history_pos.append(pso.gbest_pos)
            history_fit.append(pso.gbest_fit)
            # New generation just started → save snapshot
            if pso.iteration != old_iter:
                gen_snapshots.append(pso.positions.copy())
        else:
            history_pos.append(pso.gbest_pos)
            history_fit.append(pso.gbest_fit)
            break

    n_frames = len(gen_snapshots)

    # =====================================================================
    #  ANIMATION — particle evolution over the PV curve
    # =====================================================================
    fig_anim, ax_anim = plt.subplots(figsize=(9, 5))
    param_text = f"$N={n_particles}$, $c_1={c1}$, $c_2={c2}$, $w=[{w_min}, {w_max}]$, $v_{{step,max}}={v_step_max_frac}\\times\\Delta V$"

    ax_anim.plot(v, p, lw=1.5, color="#3a86ff", zorder=1, label="Fitness curve")
    ax_anim.axvline(v_global, color="gray", ls="--", lw=0.8, alpha=0.6,
                    label="Global optimum")
    ax_anim.axhline(p_global, color="green", ls=":", lw=0.8, alpha=0.5,
                    label=f"Global max = {p_global:.3f} mW")
    ax_anim.set_xlabel(f"V (V) \n {param_text}")
    ax_anim.set_ylabel("P (mW)")
    ax_anim.grid(True, alpha=0.3)

    # Scatter for particles & gbest marker
    particles_scatter = ax_anim.scatter(
        [], [], s=60, c="#ff006e", edgecolors="k", linewidths=0.6,
        zorder=3, label="Particles",
    )
    gbest_marker = ax_anim.scatter(
        [], [], s=140, marker="*", c="#ffbe0b", edgecolors="k",
        linewidths=0.7, zorder=4, label="gbest",
    )
    ax_anim.legend(loc="upper left", fontsize=8)

    # Trail dots (fading history)
    trail_artists = []

    def init():
        particles_scatter.set_offsets(np.empty((0, 2)))
        gbest_marker.set_offsets(np.empty((0, 2)))
        return [particles_scatter, gbest_marker]

    def update(frame):
        nonlocal trail_artists
        positions = gen_snapshots[frame]
        powers = pv_sim_curve(positions)

        # Fade previous particles as trail
        for art in trail_artists:
            art.remove()
        trail_artists.clear()

        # Draw faint trails for last few generations
        lookback = min(frame, 5)
        for k in range(1, lookback + 1):
            past = gen_snapshots[frame - k]
            alpha = 0.15 * (1 - k / (lookback + 1))
            tr = ax_anim.scatter(
                past, pv_sim_curve(past), s=20, c="#ff006e",
                alpha=alpha, zorder=2,
            )
            trail_artists.append(tr)

        # Current particles
        particles_scatter.set_offsets(np.column_stack([positions, powers]))

        # gbest — best seen so far up to this generation
        best_idx = np.argmax(powers)
        gbest_v = positions[best_idx]
        gbest_p = powers[best_idx]
        gbest_marker.set_offsets([[gbest_v, gbest_p]])

        ax_anim.set_title(
            f"PSO Particle Evolution — Gen {frame}/{n_frames - 1}",
            fontsize=10
        )
        return [particles_scatter, gbest_marker] + trail_artists

    anim = FuncAnimation(
        fig_anim, update, frames=n_frames,
        init_func=init, interval=2000, blit=False, repeat=True,
    )
    plt.tight_layout()
    plt.show()

    # =====================================================================
    #  STATIC — convergence plot (the second graph)
    # =====================================================================
    fig_conv, ax_conv = plt.subplots(figsize=(9, 3.5))
    ax_conv.plot(history_fit, "-o", markersize=4, mfc="white")
    ax_conv.axhline(p_global, color="green", ls=":", lw=0.8, alpha=0.5)
    ax_conv.set_xlabel("Hardware cycle")
    ax_conv.set_ylabel("P_gbest (mW)")
    ax_conv.set_title("PSO Convergence")
    ax_conv.grid(True)
    ax_conv.tick_params(labelsize=9)
    plt.tight_layout()
    plt.show()

    print(f"Global optimum : V≈{v_global:.4f} V, P≈{p_global:.3f} mW")
    print(f"PSO found      : V≈{history_pos[-1]:.4f} V, P≈{history_fit[-1]:.3f} mW")
    print(f"Error          : {(abs(v_global - history_pos[-1]))/v_global*100:.2f} %")

if __name__ == "__main__":
    main()