import numpy as np
import matplotlib.pyplot as plt


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
    v_min, v_max = 0.0, 1.1
    v = np.linspace(v_min, v_max, 3000)
    p = pv_sim_curve(v)

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Find global optimum for reference
    global_idx = np.argmax(p)
    v_global = v[global_idx]
    p_global = p[global_idx]

    # --- static fitness curve with global and local optima ---
    ax[0].plot(v, p, lw=1.5, label="Fitness curve")
    ax[0].axvline(v_global, color="gray", ls="--", lw=0.8,
                  alpha=0.6, label="Global optimum")
    ax[0].set_title("Multi-modal PV curve — Partial Shading Scenario")
    ax[0].set_ylabel("P (mW)")
    ax[0].grid(True)
    ax[0].legend(loc="best", fontsize=8)

    # --- PSO simulation: step-by-step tracking ---
    pso = MPPT_PSO(
        v_min=v_min, v_max=v_max,
        n_particles=12,
        max_iterations=30,
        stagnation_patience=4,
        v_step_max_frac=0.25,
        w_max=0.95, w_min=0.30,
        c1=1.5, c2=1.5,
        rng_seed=42,
    )
    history_pos = []
    history_fit = []

    for _ in range(300):
        if pso.mode == "PSO":
            fitness = float(pv_sim_curve(pso.positions[pso.particle_idx]))
            pso.step(fitness)
            history_pos.append(pso.gbest_pos)
            history_fit.append(pso.gbest_fit)
        else:
            history_pos.append(pso.gbest_pos)
            history_fit.append(pso.gbest_fit)
            break

    ax[0].plot(history_pos, [pv_sim_curve(x) for x in history_pos],
               "r.-", ms=6, mew=2, label="PSO gbest track", alpha=0.8)
    ax[0].axhline(p_global, color="green", ls=":", lw=0.8, alpha=0.5,
                  label=f"Global max = {p_global:.3f} mW")

    # --- convergence plot ---
    ax[1].plot(history_fit, "-o", markersize=4, mfc="white")
    ax[1].axhline(p_global, color="green", ls=":", lw=0.8, alpha=0.5)
    ax[1].set_xlabel("Hardware cycle")
    ax[1].set_ylabel("P_gbest (mW)")
    ax[1].set_title("PSO Convergence")
    ax[1].grid(True)
    ax[1].legend(fontsize=8)

    for a in ax:
        a.tick_params(labelsize=9)

    plt.tight_layout()
    plt.show()

    print(f"Global optimum : V≈{v_global:.4f} V, P≈{p_global:.3f} mW")
    print(f"PSO found      : V≈{history_pos[-1]:.4f} V, P≈{history_fit[-1]:.3f} mW")
    print(f"Error          : {(abs(v_global - history_pos[-1]))/v_global*100:.2f} %")

if __name__ == "__main__":
    main()