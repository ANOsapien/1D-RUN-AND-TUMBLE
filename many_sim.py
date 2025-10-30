#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

# ---------- Core Gillespie simulation for N particles ----------
def simulate_run_tumble_N(L=100, N=10, gamma=1.0, omega=0.01, t_max=200.0, seed=None):
    """
    Simulate N run-and-tumble particles on a 1D periodic lattice using a Gillespie algorithm.

    Parameters
    ----------
    L : int
        Number of lattice sites (periodic boundary conditions).
    N : int
        Number of particles (N < L).
    gamma : float
        Hop rate per particle.
    omega : float
        Tumbling (direction-reversal) rate per particle.
    t_max : float
        Total simulation time.
    seed : int or None
        RNG seed.

    Returns
    -------
    times : np.ndarray
        Recorded event times.
    positions : np.ndarray, shape (len(times), N)
        Unwrapped particle positions (can exceed L, modulo for display).
    velocities : np.ndarray, shape (len(times), N)
        Velocities (+1 or -1).
    """
    rng = np.random.default_rng(seed)

    # initial distinct lattice positions
    positions = rng.choice(np.arange(L), size=N, replace=False).astype(float)
    velocities = rng.choice([1, -1], size=N)

    t = 0.0
    times = [0.0]
    positions_hist = [positions.copy()]
    velocities_hist = [velocities.copy()]

    while t < t_max:
        # Lattice occupancy (rounded to nearest int)
        sites = np.round(positions).astype(int) % L

        # Compute allowed hops and total rates
        rates = np.zeros(2 * N)
        for i in range(N):
            target = (sites[i] + velocities[i]) % L
            # Hop if target site not occupied
            if target not in sites:
                rates[i] = gamma
            rates[N + i] = omega  # tumble always allowed

        R = rates.sum()
        if R <= 0:
            break

        # Gillespie step
        dt = -np.log(rng.random()) / R
        t += dt
        event_index = np.searchsorted(np.cumsum(rates), rng.random() * R)

        if event_index < N:
            # Hop event
            i = event_index
            positions[i] += velocities[i]
        else:
            # Tumble event
            i = event_index - N
            velocities[i] *= -1

        times.append(t)
        positions_hist.append(positions.copy())
        velocities_hist.append(velocities.copy())

    return np.array(times), np.array(positions_hist), np.array(velocities_hist), L

# ---------- Space-time plot for N particles ----------
def make_spacetime_plot(times, positions, L, savepath):
    """
    Space-time plot for N particles with correct periodic boundary handling.
    No horizontal wrap lines. All trajectories black.
    """
    N = positions.shape[1]
    fig, ax = plt.subplots(figsize=(6, 8))
    y = times

    for i in range(N):
        x_mod = np.mod(positions[:, i], L)
        x_plot = x_mod.copy()
        y_plot = y.copy()

        # Detect wrap-around between consecutive points
        jumps = np.abs(np.diff(x_mod))
        bad = np.where(jumps > L / 2)[0]

        # Insert NaN at every wrap jump
        x_plot = x_plot.astype(float)
        y_plot = y_plot.astype(float)
        for j in bad[::-1]:
            x_plot = np.insert(x_plot, j + 1, np.nan)
            y_plot = np.insert(y_plot, j + 1, np.nan)

        ax.plot(x_plot, y_plot, color="black", lw=0.8)

    ax.set_xlim(0, L)
    ax.set_ylim( 0,times.max())
    ax.set_xlabel("Position")
    ax.set_ylabel("Time")
    ax.set_title(f"Space-time plot (N={N})")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close(fig)
    print(f"Saved clean plot to {savepath}")


# ---------- Animation for N particles ----------
def make_animation(times, positions, L, savepath, fps=30):
    N = positions.shape[1]
    n_frames = 300
    t_grid = np.linspace(0, times[-1], n_frames)
    interp_positions = np.array([
        np.interp(t_grid, times, positions[:, i]) for i in range(N)
    ])

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(0, L)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel('Lattice position')
    ax.set_title(f'Run-and-Tumble Simulation (N={N})')

    colors = plt.cm.tab10(np.linspace(0, 1, N))
    points = [ax.plot([], [], 'o', color=colors[i])[0] for i in range(N)]

    def init():
        for p in points:
            p.set_data([], [])
        return points

    def update(frame):
        for i, p in enumerate(points):
            p.set_data([interp_positions[i, frame] % L], [0])
        return points

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=n_frames, interval=1000/fps, blit=True)

    if savepath.endswith(".gif"):
        anim.save(savepath, writer='pillow', fps=fps)
    else:
        anim.save(savepath, writer='ffmpeg', fps=fps)
    plt.close(fig)
    print(f"Saved animation to {savepath}")

# ---------- Parameter test ----------
def test_N_particles():
    L = 300
    N = 60
    gamma = 1.0
    omega = 0.5
    t_max = 200
    seed = 42
    os.makedirs("outputs", exist_ok=True)

    times, positions, velocities, L = simulate_run_tumble_N(
        L=L, N=N, gamma=gamma, omega=omega, t_max=t_max, seed=seed
    )

    make_spacetime_plot(times, positions, L, "outputs/spacetime_OMEGA{omega}_L{L}55.png")
    make_animation(times, positions, L, "outputs/animation_OMEGA{omega}_L{L}55.gif", fps=30)

if __name__ == "__main__":
    test_N_particles()
