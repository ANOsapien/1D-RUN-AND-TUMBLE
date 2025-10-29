#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 12})

def simulate_two_run_tumble(L=100, gamma=1.0, omega=0.01, t_max=200.0, seed=None, 
                            record_every_event=True):
    """
    Gillespie / event-driven simulation.

    Parameters
    ----------
    L : int
        Number of lattice sites (periodic).
    gamma : float
        Hop rate per particle (attempt rate to move one site in its orientation).
    omega : float
        Velocity reversal rate per particle (this code uses omega = alpha/2 in paper notation).
    t_max : float
        Simulation end time.
    seed : int or None
        RNG seed.
    record_every_event : bool
        If True, record state at each event (recommended for plotting).

    Returns
    -------
    times : np.array  (N,)
        Times at which states were recorded (starting at 0).
    pos1, pos2 : np.array  (N,)
        Unwrapped particle positions (can be wrapped for display).
    vel1, vel2 : np.array  (N,)
        Orientations (+1 or -1).
    """
    rng = np.random.default_rng(seed)
    # initialize positions and velocities
    # choose two distinct lattice sites
    p = rng.choice(np.arange(L), size=2, replace=False)
    # ensure p[0] < p[1] for nicer initial separation (not required)
    if p[0] > p[1]:
        p = p[::-1]
    # unwrapped real-space positions (start in [0,L) but we will allow them to move beyond boundaries)
    x_real = p.astype(float)  # unwrapped coordinates for plotting continuous lines
    v = rng.choice([1, -1], size=2)  # +1 right, -1 left, equal prob

    t = 0.0
    times = [0.0]
    pos1 = [x_real[0]]
    pos2 = [x_real[1]]
    vel1 = [v[0]]
    vel2 = [v[1]]

    # event loop
    while t < t_max:
        # compute rates for each possible event:
        rates = []

        # particle 0 hop rate: only if target site not occupied (in lattice sense)
        # compute target lattice site for current particle positions (rounded to int mod L)
        site0 = int(np.round(x_real[0])) % L
        site1 = int(np.round(x_real[1])) % L

        # target sites for hops (on lattice)
        target0 = (site0 + v[0]) % L
        target1 = (site1 + v[1]) % L

        # hop allowed only if target is not occupied (hard-core exclusion)
        hop0_allowed = (target0 != site1)
        hop1_allowed = (target1 != site0)

        # rates:
        r_hop0 = gamma if hop0_allowed else 0.0
        r_hop1 = gamma if hop1_allowed else 0.0
        r_tumble0 = omega
        r_tumble1 = omega

        rates = np.array([r_hop0, r_hop1, r_tumble0, r_tumble1])
        R = rates.sum()
        if R <= 0:
            # no events possible (shouldn't happen), break
            break

        # Gillespie time-step
        u = rng.random()
        dt = -np.log(u) / R
        t += dt

        # choose which event occurs
        u2 = rng.random() * R
        cum = np.cumsum(rates)
        event_idx = int(np.searchsorted(cum, u2, side='right'))

        # perform the event
        if event_idx == 0:
            # hop particle 0
            # increment unwrapped position by +1 or -1
            x_real[0] += v[0]
        elif event_idx == 1:
            # hop particle 1
            x_real[1] += v[1]
        elif event_idx == 2:
            # tumble particle 0 -> reverse sign
            v[0] *= -1
        elif event_idx == 3:
            # tumble particle 1
            v[1] *= -1

        # record
        if record_every_event:
            times.append(t)
            pos1.append(x_real[0])
            pos2.append(x_real[1])
            vel1.append(v[0])
            vel2.append(v[1])

    return np.array(times), np.array(pos1), np.array(pos2), np.array(vel1), np.array(vel2), L

def make_spacetime_plot(times, x1, x2, v1, v2, L, figsize=(6,8), show=True, savepath=None):
    """
    Create space-time plot (time vertical downward = negative y axis).
    x1,x2 are unwrapped positions. We'll map to [0,L) for x display but break lines on wraps.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, L)
    ax.set_ylim(times.max(), 0)   # time negative downward
    ax.set_xlabel('Position (lattice site)')
    ax.set_ylabel('Time')
    ax.set_title('Space-time plot')

    # helper to prepare continuous segments for plotting; insert NaNs to break lines at periodic wrap
    def prepare_segments(x_unwrapped):
        """Return x_display and y values with NaNs inserted where wrapping occurs"""
        x_display = (x_unwrapped % L).copy()
        y = times.copy()
        # detect large jumps in unwrapped trajectory (wrap events)
        dx = np.diff(x_unwrapped)
        # whenever abs(dx) > L/2 it's a wrap (we treat this as a discontinuity)
        wrap_idx = np.where(np.abs(dx) > (L/2.0))[0]
        if len(wrap_idx) == 0:
            return x_display, y
        # build new arrays inserting NaNs after indices in wrap_idx
        xs = []
        ys = []
        N = len(x_unwrapped)
        insert_set = set(wrap_idx.tolist())
        for i in range(N):
            xs.append(x_display[i])
            ys.append(y[i])
            if i in insert_set:
                # insert break
                xs.append(np.nan)
                ys.append(np.nan)
        return np.array(xs), np.array(ys)

    x1_disp, y1 = prepare_segments(x1)
    x2_disp, y2 = prepare_segments(x2)

    # draw trajectories
    # style: particle 1 solid, particle 2 dotted; linewidth tuned; orientation influences alpha or linewidth optionally
    ax.plot(x1_disp, y1, lw=1.6, linestyle='-', label='particle 1')
    ax.plot(x2_disp, y2, lw=1.6, linestyle='--', label='particle 2')

    # mark collision / jammed events where sites are adjacent and facing each other
    # recompute lattice sites from unwrapped
    sites1 = (np.round(x1) % L).astype(int)
    sites2 = (np.round(x2) % L).astype(int)
    # jam condition: adjacent (difference 1 mod L) and velocities opposite directed at each other
    jam_times = []
    jam_x = []
    for i in range(len(times)):
        s1 = sites1[i]
        s2 = sites2[i]
        # separation on ring
        sep = (s2 - s1) % L
        # check adjacent both sides
        adj12 = (sep == 1)
        adj21 = (sep == L-1)
        if adj12 and v1[i]==+1 and v2[i]==-1:
            jam_times.append(times[i])
            jam_x.append((x1[i] % L + 0.0))
        elif adj21 and v1[i]==-1 and v2[i]==+1:
            jam_times.append(times[i])
            jam_x.append((x2[i] % L + 0.0))

    if len(jam_times) > 0:
        ax.scatter(jam_x, -np.array(jam_times), marker='o', s=18, color='k', label='jammed (adjacent)', zorder=5)

    ax.invert_yaxis()  # optional: time downward visually
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {savepath}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # parameters (similar to Fig.1 in the paper)
    L = 100             # lattice size
    gamma = 1.0          # hop rate (set to 1 by rescaling time)
    # paper uses omega := alpha/(2*gamma); in that notation alpha is tumbling rate
    # Here we take omega=0.01 to get low tumble rate (long runs)
    omega = 500
    t_max = 100.0
    seed = 42

    times, x1, x2, v1, v2, L = simulate_two_run_tumble(L=L, gamma=gamma, omega=omega, t_max=t_max, seed=seed)
    make_spacetime_plot(times, x1, x2, v1, v2, L, figsize=(5,9), savepath="spacetime_2pt.png")
