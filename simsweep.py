#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

# ---------- Core Gillespie simulation ----------
def simulate_two_run_tumble(L=1000, gamma=1.0, omega=0.01, t_max=150.0, seed=None):
    rng = np.random.default_rng(seed)
    p = rng.choice(np.arange(L), size=2, replace=False)
    if p[0] > p[1]:
        p = p[::-1]
    x_real = p.astype(float)
    v = rng.choice([1, -1], size=2)
    t = 0.0

    times = [0.0]
    pos1 = [x_real[0]]
    pos2 = [x_real[1]]
    vel1 = [v[0]]
    vel2 = [v[1]]

    while t < t_max:
        site0 = int(round(x_real[0])) % L
        site1 = int(round(x_real[1])) % L
        target0 = (site0 + v[0]) % L
        target1 = (site1 + v[1]) % L

        hop0_allowed = (target0 != site1)
        hop1_allowed = (target1 != site0)

        r_hop0 = gamma if hop0_allowed else 0.0
        r_hop1 = gamma if hop1_allowed else 0.0
        r_tumble0 = omega
        r_tumble1 = omega

        rates = np.array([r_hop0, r_hop1, r_tumble0, r_tumble1])
        R = rates.sum()
        if R <= 0:
            break

        dt = -np.log(rng.random()) / R
        t += dt
        event = np.searchsorted(np.cumsum(rates), rng.random() * R)

        if event == 0: x_real[0] += v[0]
        elif event == 1: x_real[1] += v[1]
        elif event == 2: v[0] *= -1
        elif event == 3: v[1] *= -1

        times.append(t)
        pos1.append(x_real[0])
        pos2.append(x_real[1])
        vel1.append(v[0])
        vel2.append(v[1])

    return np.array(times), np.array(pos1), np.array(pos2), np.array(vel1), np.array(vel2), L

# ---------- Space-time plot ----------
def make_spacetime_plot(times, x1, x2, L, savepath):
    fig, ax = plt.subplots(figsize=(5,8))
    y = -times
    x1_mod = (x1 % L)
    x2_mod = (x2 % L)

    ax.plot(x1_mod, y, '-', lw=1.5, label='particle 1')
    ax.plot(x2_mod, y, '--', lw=1.5, label='particle 2')
    ax.set_xlim(0, L)
    ax.set_ylim(-times.max(), 0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Time (downward)')
    ax.set_title('Space-time plot')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close(fig)
    print(f"Saved {savepath}")

# ---------- Animation ----------
def make_animation(times, x1, x2, L, savepath, fps=30):
    # interpolate positions to fixed time grid for smooth animation
    n_frames = 300
    t_grid = np.linspace(0, times[-1], n_frames)
    x1i = np.interp(t_grid, times, x1)
    x2i = np.interp(t_grid, times, x2)

    fig, ax = plt.subplots(figsize=(6,2))
    ax.set_xlim(0, L)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel('Lattice position')
    ax.set_title('Run-and-Tumble Motion (animation)')

    p1, = ax.plot([], [], 'ro', markersize=10, label='particle 1')
    p2, = ax.plot([], [], 'bo', markersize=10, label='particle 2')
    ax.legend()

    def init():
        p1.set_data([], [])
        p2.set_data([], [])
        return p1, p2

    def update(i):
        p1.set_data([x1i[i] % L], [0.2])
        p2.set_data([x2i[i] % L], [-0.2])
        return p1, p2

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=n_frames, interval=1000/fps, blit=True)

    if savepath.endswith(".gif"):
        anim.save(savepath, writer='pillow', fps=fps)
    else:
        anim.save(savepath, writer='ffmpeg', fps=fps)
    plt.close(fig)
    print(f"Saved animation to {savepath}")

# ---------- Parameter sweep ----------
def parameter_sweep():
    L = 100
    gamma = 1.0
    t_max = 300
    seed = 42
    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    omega_values = [0.001, 0.01, 0.05, 0.1]
    for omega in omega_values:
        print(f"Running ω={omega} ...")
        times, x1, x2, v1, v2, L = simulate_two_run_tumble(
            L=L, gamma=gamma, omega=omega, t_max=t_max, seed=seed)
        make_spacetime_plot(times, x1, x2, L, savepath=f"{outdir}/spacetime_omega{omega:.3f}.png")
        make_animation(times, x1, x2, L, savepath=f"{outdir}/animation_omega{omega:.3f}.mp4")

if __name__ == "__main__":
    parameter_sweep()
