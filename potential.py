#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Analytic formulas (from PRL)
# -------------------------
def analytic_P_pp_pm(n_array, L, omega):
    z = 1.0 + omega - np.sqrt(omega * (2.0 + omega))
    p = 1.0 - z**2
    p0 = (1.0 - z) / (1.0 + z) * p
    q = (1.0 - z)**2 * (1.0 - z**L)
    Delta = 2.0 * (1.0 + z) * (z - z**L)
    Z = 4.0 * (2.0 * (1.0 + z) * (z - z**L) + (L - 1) * q)
    ns = n_array.astype(np.int64)
    zns = z**ns
    zL_minus_n = z**(L - ns)
    Ppp = (p * (zns + zL_minus_n) + q) / Z
    Ppm = (p0 * (zns - zL_minus_n) + q) / Z
    Ppm[0] += Delta / Z
    return Ppp, Ppm

# -------------------------
# Cyclic Gaussian smoothing via FFT convolution
# -------------------------
def cyclic_gaussian_smooth(P, sigma):
    """
    Periodic (cyclic) Gaussian smoothing of array P using FFT.
    sigma: std in bins.
    """
    M = len(P)
    k = np.fft.fftfreq(M)  # frequencies (cycles per sample)
    fft_kernel = np.exp(-2.0 * (np.pi**2) * (sigma**2) * (k**2))
    fft_P = np.fft.fft(P)
    P_smooth = np.fft.ifft(fft_P * fft_kernel).real
    # sanitize & renormalize to original sum
    P_smooth[P_smooth < 0] = 0.0
    s_old = P.sum()
    s_new = P_smooth.sum()
    if s_new > 0:
        P_smooth *= (s_old / s_new)
    return P_smooth

# -------------------------
# Gillespie simulation that accumulates TIME in each (sector, n) bin
# -------------------------
def simulate_two_run_tumble_time_weighted(L=100, gamma=1.0, omega=0.01,
                                          t_max=2e6, seed=None,
                                          burn_time=2e5):
    """
    Gillespie simulation that returns time-in-bin counts (float) for each sector and separation n=1..L-1.
    Bins indexed as [sector, n-1] with sector mapping:
      0: ++
      1: +-
      2: -+
      3: --
    Time weighting: for each step we draw dt, and we add the overlap of [t, t+dt) with [burn_time, t_max)
    to the current (sector,n) bin before applying the event.
    """
    rng = np.random.default_rng(seed)
    # initial distinct sites
    p = rng.choice(np.arange(L), size=2, replace=False)
    if p[0] > p[1]:
        p = p[::-1]
    x_real = p.astype(float)
    v = rng.choice([1, -1], size=2)
    t = 0.0
    time_in_bins = np.zeros((4, L-1), dtype=float)
    total_time_counted = 0.0

    while t < t_max:
        # compute current lattice sites and event rates
        site0 = int(np.round(x_real[0])) % L
        site1 = int(np.round(x_real[1])) % L
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

        # draw dt for current state
        dt = -np.log(rng.random()) / R
        t_next = t + dt

        # compute overlap of [t, t_next) with [burn_time, t_max)
        # overlap_start = max(t, burn_time), overlap_end = min(t_next, t_max)
        overlap_start = t if t > burn_time else burn_time
        overlap_end = t_next if t_next < t_max else t_max
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > 0:
            # determine current sector and separation n (based on integer lattice rounding of positions)
            s1 = int(np.round(x_real[0])) % L
            s2 = int(np.round(x_real[1])) % L
            n = (s2 - s1) % L
            if n != 0:
                n_idx = n - 1
                if v[0] == +1 and v[1] == +1:
                    si = 0
                elif v[0] == +1 and v[1] == -1:
                    si = 1
                elif v[0] == -1 and v[1] == +1:
                    si = 2
                else:
                    si = 3
                time_in_bins[si, n_idx] += overlap
                total_time_counted += overlap

        # advance time and perform the sampled event (state change)
        t = t_next
        # choose event
        u2 = rng.random() * R
        event_idx = int(np.searchsorted(np.cumsum(rates), u2, side='right'))
        if event_idx == 0:
            x_real[0] += v[0]
        elif event_idx == 1:
            x_real[1] += v[1]
        elif event_idx == 2:
            v[0] *= -1
        elif event_idx == 3:
            v[1] *= -1

    return time_in_bins, total_time_counted

# -------------------------
# Plot helper: form probabilities, smooth, log, plot up to n_max
# -------------------------
def plot_pair_potentials_time_weighted(time_in_bins, total_time, L, omega,
                                       pseudocount=1e-12, analytic_sigma=1.5,
                                       n_max_plot=50, eps_log=1e-16, savepath=None):
    """
    time_in_bins: array shape (4, L-1) giving total time spent in each (sector,n)
    total_time: total time accumulated (should equal sum(time_in_bins))
    """
    # Joint probability P_{sector,n} = time_in_bins / total_time
    P_joint = time_in_bins / float(total_time + 1e-30)

    # optionally apply tiny pseudocount to avoid exact zeros for log (but with time-weighting exact zeros mean prob ~0)
    # add pseudocount uniformly (small) then renormalize
    if pseudocount is not None and pseudocount > 0.0:
        P_joint += pseudocount
        P_joint /= P_joint.sum()

    Ppp = P_joint[0, :].copy()   # ++
    Ppm = P_joint[1, :].copy()   # +-

    # analytic exact & smooth
    n_array = np.arange(1, L)
    Ppp_ex, Ppm_ex = analytic_P_pp_pm(n_array, L, omega)
    Ppp_ex_s = cyclic_gaussian_smooth(Ppp_ex, sigma=analytic_sigma)
    Ppm_ex_s = cyclic_gaussian_smooth(Ppm_ex, sigma=analytic_sigma)

    # smooth simulation histograms a bit for visual comparison (optional)
    Ppp_s = cyclic_gaussian_smooth(Ppp, sigma=analytic_sigma)
    Ppm_s = cyclic_gaussian_smooth(Ppm, sigma=analytic_sigma)

    Upp_sim = -np.log(Ppp_s + eps_log)
    Upm_sim = -np.log(Ppm_s + eps_log)
    Upp_ex = -np.log(Ppp_ex_s + eps_log)
    Upm_ex = -np.log(Ppm_ex_s + eps_log)

    n_vals = np.arange(1, L)
    mask = n_vals <= n_max_plot

    plt.figure(figsize=(8,5))
    plt.plot(n_vals[mask], Upp_ex[mask], lw=1.5, alpha=0.9, label=r" $-\ln P_{++}(n)$")
    plt.plot(n_vals[mask], Upm_ex[mask], lw=1.5, linestyle='--', alpha=0.9, label=r"$-\ln P_{+-}(n)$")
    plt.xlabel("Separation $n$ (sites)")
    plt.ylabel(r"Effective potential $U(n)=-\ln P(n)$")
    plt.title(f"Effective potentials (L={L}, omega={omega})")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
        print("Saved", savepath)
    plt.show()

# -------------------------
# Example run
# -------------------------
if __name__ == "__main__":
    L = 100
    omega = 0.01
    # run the time-weighted sim (this can take a while depending on t_max)
    time_in_bins, total_time = simulate_two_run_tumble_time_weighted(L=L, gamma=1.0,
                                                                     omega=omega,
                                                                     t_max=2e6,
                                                                     seed=12345,
                                                                     burn_time=2e5)
    print("Total time counted (should approx t_max - burn_time):", total_time)
    plot_pair_potentials_time_weighted(time_in_bins, total_time, L, omega,
                                       pseudocount=1e-12, analytic_sigma=1.8,
                                       n_max_plot=50, savepath="pair_potentials_timeweighted.png")
