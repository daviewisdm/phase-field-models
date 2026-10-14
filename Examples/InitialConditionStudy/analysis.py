"""Metrics and figures for the initial-condition study.

Usage:  python analysis.py --out results        (directory written by run_study.py)

Reads <out>/manifest.json, <out>/ics/*.npy and, per profile, <out>/runs/<key>/{trajectory_0.dat,last_0.dat,energy.dat}
(formats documented in REPO_NOTES.md). Writes PNG figures, metrics_summary.csv/.png, metrics.json and per-frame
time series (timeseries_<key>.csv) into <out>.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import initial_conditions as ic  # noqa: E402

DPI = 160
CMAP = "RdBu_r"
ONSET_STD = 0.1          # onset: RMS deviation of rho reaches 10x the common initial RMS (0.01)
SAT_FRAC = 0.9           # saturation: RMS deviation reaches 90% of its final value
T_LABEL = "t [simulation time units]"
L_LABEL = "[grid units, dx = 1]"


# ----------------------------------------------------------------------------------------------- loading
def load_run(run_dir: Path, dt: float) -> dict:
    """Trajectory frames (trajectory_0.dat + last_0.dat appended) and energy.dat of one run."""
    steps, _, frames = ic.read_frames(run_dir / "trajectory_0.dat")
    ls, _, lf = ic.read_frames(run_dir / "last_0.dat")
    if ls[0] > steps[-1]:
        steps, frames = np.append(steps, ls), np.concatenate([frames, lf])
    return {"step": steps, "t": steps * dt, "rho": frames, "energy": ic.read_energy(run_dir / "energy.dat")}


# ----------------------------------------------------------------------------------------------- metrics
def wavenumbers(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Angular wavenumbers (k_i, k_j) of an N x N FFT grid (dx = 1), indexing='ij'."""
    k = 2 * np.pi * np.fft.fftfreq(N)
    return np.meshgrid(k, k, indexing="ij")


def structure_factor(rho: np.ndarray) -> np.ndarray:
    """S(k) = |FFT(rho - mean)|^2 / N^2 for frames (F, N, N)."""
    d = rho - rho.mean(axis=(-2, -1), keepdims=True)
    return np.abs(np.fft.fft2(d)) ** 2 / (rho.shape[-1] ** 2)


def radial_average(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shell-average S over integer shells n = round(|k| N / 2pi), n = 1..N/2. Returns (k_shell, S_shell)."""
    N = S.shape[-1]
    ki, kj = wavenumbers(N)
    n = np.rint(np.sqrt(ki**2 + kj**2) * N / (2 * np.pi)).astype(int)
    shells = np.arange(1, N // 2 + 1)
    Sr = np.stack([S[..., n == s].mean(axis=-1) for s in shells], axis=-1)
    return 2 * np.pi * shells / N, Sr


def periodic_count(mask: np.ndarray) -> int:
    """Number of 4-connected components of a boolean 2D mask with periodic boundaries."""
    lab, n = ndimage.label(mask)
    parent = np.arange(n + 1)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a_edge, b_edge in ((lab[0, :], lab[-1, :]), (lab[:, 0], lab[:, -1])):
        for a, b in zip(a_edge, b_edge):
            if a and b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return len({find(a) for a in range(1, n + 1)})


def frame_metrics(rho: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame scalar metrics for frames of shape (F, N, N)."""
    N = rho.shape[-1]
    S = structure_factor(rho)
    ki, kj = wavenumbers(N)
    kmag = np.sqrt(ki**2 + kj**2)
    nz = kmag > 0
    Stot = S[:, nz].sum(axis=1)
    k1 = (S[:, nz] * kmag[nz]).sum(axis=1) / Stot
    k_shell, Sr = radial_average(S)
    pos = rho > 0
    bonds = (pos != np.roll(pos, 1, axis=1)).sum(axis=(1, 2)) + (pos != np.roll(pos, 1, axis=2)).sum(axis=(1, 2))
    aniso = (S * (ki**2 - kj**2)).sum(axis=(1, 2)) / (S * kmag**2).sum(axis=(1, 2))
    return {
        "mean": rho.mean(axis=(1, 2)), "std": rho.std(axis=(1, 2)),
        "min": rho.min(axis=(1, 2)), "max": rho.max(axis=(1, 2)),
        "L1": 2 * np.pi / k1, "k_peak": k_shell[Sr.argmax(axis=1)],
        "phase_frac_pos": pos.mean(axis=(1, 2)),
        "interface_len": bonds * np.pi / 4,  # Cauchy-Crofton correction of lattice crossing-bond count
        "anisotropy": aniso,
        "n_dom_pos": np.array([periodic_count(p) for p in pos]),
        "n_dom_neg": np.array([periodic_count(~p) for p in pos]),
    }


def crossing_time(t: np.ndarray, y: np.ndarray, level: float, log: bool = True) -> float:
    """First time y crosses `level` upward, interpolated linearly (in log y if log=True). NaN if never."""
    idx = np.nonzero((y[1:] >= level) & (y[:-1] < level))[0]
    if len(idx) == 0:
        return float(t[0]) if y[0] >= level else float("nan")
    a = idx[0]
    f = (lambda v: np.log(v)) if log else (lambda v: v)
    w = (f(level) - f(y[a])) / (f(y[a + 1]) - f(y[a]))
    return float(t[a] + w * (t[a + 1] - t[a]))


def linear_theory(rho0: np.ndarray, cfg: dict, t: np.ndarray) -> tuple[np.ndarray, dict]:
    """Linearised explicit-Euler prediction of RMS(rho - mean) vs t from the IC alone.

    Per step each Fourier mode is multiplied by g = 1 - dt*M*lam*(f''(rho_mean) + c*lam), with lam the eigenvalue of
    -Laplacian for the repo's 5-point stencil and c = interface_scalar*k.
    """
    N, dt, M = rho0.shape[-1], cfg["dt"], cfg["M"]
    c = cfg["interface_scalar"] * cfg["k"]
    eps = cfg["landau"]["epsilon"]
    rm = rho0.mean()
    fpp = -eps + 3 * rm**2
    n = np.fft.fftfreq(N) * N
    ni, nj = np.meshgrid(n, n, indexing="ij")
    lam = 4 * (np.sin(np.pi * ni / N) ** 2 + np.sin(np.pi * nj / N) ** 2) / cfg["dx"] ** 2
    g = 1 - dt * M * lam * (fpp + c * lam)
    assert np.all(np.abs(g) < 1 + 1e-3), "explicit Euler amplification > 1 beyond physical growth"
    P = np.abs(np.fft.fft2(rho0 - rm)) ** 2 / N**4  # sum(P) = variance
    logg = np.log(np.abs(g) + 1e-300)
    with np.errstate(divide="ignore"):
        logP = np.log(P)
    logstd = []
    for tt in t:
        a = logP + 2 * logg * np.rint(tt / dt)
        amax = a.max()
        logstd.append(0.5 * (amax + np.log(np.exp(a - amax).sum())))
    std = np.exp(np.minimum(np.array(logstd), np.log(1e3)))
    sigma_cont = -lam * (fpp + c * lam)  # continuous-time growth rate per mode (M = 1)
    info = {"fpp": fpp, "q_max": float(np.sqrt(-fpp / (2 * c))), "sigma_max": float(fpp**2 / (4 * c) * M),
            "lambda_max": float(2 * np.pi / np.sqrt(-fpp / (2 * c))), "q_c": float(np.sqrt(-fpp / c)),
            "lam": lam, "sigma": sigma_cont * M, "P": P}
    return std, info


def fit_coarsening(t: np.ndarray, L: np.ndarray, t_start: float, L_max: float) -> tuple[float, float, float, float]:
    """Fit log L = n log t + c from t_start until L first reaches L_max (finite-size cut), on log-spaced frames.

    Returns (n, c, t0, t1); n is NaN if fewer than 5 frames or less than a factor 3 in time are available.
    """
    after = t >= t_start
    hit = np.nonzero(after & (L >= L_max))[0]
    t_end = t[hit[0]] if len(hit) else t[-1]
    sel = after & (t < t_end if len(hit) else t <= t_end) & (t > 0)
    if sel.sum() < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")
    ts, Ls = t[sel], L[sel]
    if ts[-1] / ts[0] < 3:
        return float("nan"), float("nan"), float(ts[0]), float(ts[-1])
    grid = np.unique(np.searchsorted(ts, np.geomspace(ts[0], ts[-1], 40)).clip(0, len(ts) - 1))
    n, c = np.polyfit(np.log(ts[grid]), np.log(Ls[grid]), 1)
    return float(n), float(c), float(ts[0]), float(ts[-1])


# ----------------------------------------------------------------------------------------------- plotting
def nearest_frames(t: np.ndarray, targets: np.ndarray) -> list[int]:
    return [int(np.abs(t - x).argmin()) for x in targets]


def plot_initial_conditions(ics: dict[str, np.ndarray], names: dict, gc: tuple[int, int], out: Path) -> None:
    keys = list(ics)
    vmax = max(np.abs(r[0] - r[0].mean()).max() for r in ics.values())
    rm = next(iter(ics.values()))[0].mean()
    fig = plt.figure(figsize=(3.2 * len(keys) + 1, 6.6))
    gs = fig.add_gridspec(2, len(keys) + 1, width_ratios=[1] * len(keys) + [0.06], height_ratios=[1, 0.8])
    for c, k in enumerate(keys):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(ics[k][0], cmap=CMAP, vmin=rm - vmax, vmax=rm + vmax, origin="upper")
        ax.axvline(gc[1], color="k", lw=0.8, ls="--")
        ax.set_title(names[k], fontsize=9)
        ax.set_xlabel(f"j {L_LABEL}", fontsize=8)
        ax.set_ylabel(f"i {L_LABEL}", fontsize=8)
        ax.tick_params(labelsize=7)
        if k == "D_gaussian":
            ax.plot(gc[1], gc[0], "k+", ms=8)
        ax2 = fig.add_subplot(gs[1, c])
        ax2.plot(np.arange(ics[k].shape[1]), ics[k][0][:, gc[1]], color=f"C{c}")
        ax2.axhline(rm, color="gray", lw=0.6)
        ax2.set_ylim(rm - vmax * 1.05, rm + vmax * 1.05)
        ax2.set_xlabel(f"i {L_LABEL}", fontsize=8)
        ax2.set_ylabel(r"$\rho$ along column j = %d" % gc[1], fontsize=8)
        ax2.tick_params(labelsize=7)
    cax = fig.add_subplot(gs[0, -1])
    fig.colorbar(im, cax=cax, label=r"$\rho$")
    fig.suptitle(r"Initial profiles (identical mean $\bar\rho$ = %.3f and RMS($\rho-\bar\rho$) = %.3g); "
                 "dashed line = location of the 1D cut" % (rm, np.sqrt(np.mean((ics[keys[0]] - rm) ** 2))), fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "initial_conditions.png", dpi=DPI)
    plt.close(fig)


def plot_snapshots(runs: dict, names: dict, times: np.ndarray, out: Path) -> None:
    keys = list(runs)
    for k in keys:
        r = runs[k]
        idx = nearest_frames(r["t"], times)
        fig, axes = plt.subplots(1, len(idx), figsize=(2.6 * len(idx) + 0.8, 3.0), constrained_layout=True)
        for ax, f in zip(axes, idx):
            im = ax.imshow(r["rho"][f], cmap=CMAP, vmin=-1, vmax=1)
            ax.set_title(f"t = {r['t'][f]:g}", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
        axes[0].set_ylabel(f"i {L_LABEL}", fontsize=8)
        fig.colorbar(im, ax=axes, label=r"$\rho$", shrink=0.85)
        fig.suptitle(f"{names[k]}: evolution of $\\rho$ (N = {r['rho'].shape[-1]}, periodic)", fontsize=10)
        fig.savefig(out / f"snapshots_{k}.png", dpi=DPI)
        plt.close(fig)

    idx0 = nearest_frames(runs[keys[0]]["t"], times)
    fig, axes = plt.subplots(len(keys), len(idx0), figsize=(2.3 * len(idx0) + 1, 2.3 * len(keys) + 0.6),
                             constrained_layout=True)
    for row, k in enumerate(keys):
        r = runs[k]
        for col, f in enumerate(nearest_frames(r["t"], times)):
            ax = axes[row, col]
            im = ax.imshow(r["rho"][f], cmap=CMAP, vmin=-1, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t = {r['t'][f]:g}", fontsize=10)
            if col == 0:
                ax.set_ylabel(ic.SHORT_NAMES.get(k, k), fontsize=10)
    fig.colorbar(im, ax=axes, label=r"$\rho$", shrink=0.6)
    fig.suptitle("All profiles (rows) at common times (columns)", fontsize=12)
    fig.savefig(out / "snapshots_all.png", dpi=DPI)
    plt.close(fig)


def write_gif(r: dict, name: str, path: Path, n_frames: int = 60) -> None:
    t = r["t"]
    idx = sorted(set(nearest_frames(t, np.concatenate([[0], np.geomspace(max(t[1], 1e-9), t[-1], n_frames - 1)]))))
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    im = ax.imshow(r["rho"][0], cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])
    title = ax.set_title("")

    def update(f: int):
        im.set_data(r["rho"][f])
        title.set_text(f"{name.split(':')[0]}  t = {t[f]:g}")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=idx, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=8), dpi=90)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(HERE / "results"))
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    man = json.loads((out / "manifest.json").read_text())
    cfg, keys, names = man["frozen"], man["profiles"], man["names"]
    N, dt = int(cfg["N"]), float(cfg["dt"])
    gc = tuple(man["gauss_center_ij"])
    eps = cfg["landau"]["epsilon"]
    rho_mean, amp = man["study"]["rho_mean"], man["study"]["amp_rms"]

    ics = {k: np.load(out / "ics" / f"{k}.npy") for k in keys}
    ic.check_fairness(ics, rho_mean, amp)
    runs = {k: load_run(out / "runs" / k, dt) for k in keys}

    # ---- integrity checks: first frame equals the IC; orientation of the off-centre Gaussian
    checks: dict[str, object] = {}
    for k in keys:
        assert np.array_equal(runs[k]["rho"][0].astype(np.float32), ics[k][0].astype(np.float32)), f"{k}: frame0 != IC"
    checks["frame0_equals_ic"] = True
    if "D_gaussian" in runs:
        f0 = runs["D_gaussian"]["rho"][0]
        peak = tuple(int(v) for v in np.unravel_index(f0.argmax(), f0.shape))
        assert peak == gc, f"Gaussian peak at {peak}, intended {gc}: orientation mismatch"
        checks["gaussian_peak_frame0_ij"] = list(peak)
        checks["gaussian_intended_ij"] = list(gc)
        checks["gaussian_transposed_ij_differs"] = gc[::-1] != gc

    # ---- metrics
    lam_theory, metrics, series = {}, {}, {}
    for k in keys:
        r = runs[k]
        m = frame_metrics(r["rho"])
        series[k] = m
        t = r["t"]
        e = r["energy"]
        std_lin, info = linear_theory(ics[k][0], cfg, t)
        lam_theory[k] = (std_lin, info)
        t_on = crossing_time(t, m["std"], ONSET_STD)
        t_on_lin = crossing_time(t, std_lin, ONSET_STD)
        # Saturation and coarsening are only defined if the profile actually phase-separated.
        t_sat = crossing_time(t, m["std"], SAT_FRAC * m["std"][-1]) if np.isfinite(t_on) else float("nan")
        F0, Fend = e["F"][0], e["F"][-1]
        t_F50 = crossing_time(e["t"], -(e["F"] - F0), 0.5 * (F0 - Fend), log=False)
        n_fit, c_fit, tf0, tf1 = fit_coarsening(t, m["L1"], 2 * t_on if np.isfinite(t_on) else np.inf, N / 2)
        drift = m["mean"] - m["mean"][0]
        metrics[k] = {
            "profile": names[k],
            "t_onset (RMS>=0.1)": t_on,
            "t_onset linear theory": t_on_lin,
            "t_sat (RMS>=0.9 final)": t_sat,
            "t_F50 (half energy drop)": t_F50,
            "F_final": float(Fend),
            "RMS_final": float(m["std"][-1]),
            "min_final": float(m["min"][-1]), "max_final": float(m["max"][-1]),
            "L1_final": float(m["L1"][-1]), "k_peak_final": float(m["k_peak"][-1]),
            "phase_frac_pos_final": float(m["phase_frac_pos"][-1]),
            "interface_len_final": float(m["interface_len"][-1]),
            "n_domains_pos_final": int(m["n_dom_pos"][-1]), "n_domains_neg_final": int(m["n_dom_neg"][-1]),
            "anisotropy_final": float(m["anisotropy"][-1]),
            "coarsening_exponent": n_fit, "fit_t_range": [tf0, tf1],
            "mass_drift_final": float(drift[-1]), "mass_drift_max_abs": float(np.abs(drift).max()),
            "energy_dat_mass_range": [float(e["mass"].min()), float(e["mass"].max())],
            "energy_monotone_violations": int((np.diff(e["F"]) > 1e-8).sum()),
        }
        with open(out / f"timeseries_{k}.csv", "w", newline="") as f:
            w = csv.writer(f)
            cols = list(m)
            w.writerow(["step", "t"] + cols + ["rms_linear_theory"])
            for i in range(len(t)):
                w.writerow([r["step"][i], t[i]] + [m[c][i] for c in cols] + [std_lin[i]])
        if k == "D_gaussian":
            fi = int(np.abs(t - t_on).argmin()) if np.isfinite(t_on) else 0
            d2 = (r["rho"][fi] - r["rho"][fi].mean()) ** 2
            ang = 2 * np.pi * np.arange(N) / N
            ci = (np.angle((d2.sum(1) * np.exp(1j * ang)).sum()) % (2 * np.pi)) * N / (2 * np.pi)
            cj = (np.angle((d2.sum(0) * np.exp(1j * ang)).sum()) % (2 * np.pi)) * N / (2 * np.pi)
            dist = float(ic.periodic_distance(N, ci, cj)[gc])
            checks["gaussian_onset_frame_t"] = float(t[fi])
            checks["gaussian_onset_centroid_ij"] = [round(float(ci), 2), round(float(cj), 2)]
            checks["gaussian_onset_centroid_dist_to_seed"] = round(dist, 2)

    info0 = lam_theory[keys[0]][1]
    theory = {"rho_mean": rho_mean, "f''(rho_mean)": info0["fpp"], "spinodal_|rho|": float(np.sqrt(eps / 3)),
              "binodal_|rho|": float(np.sqrt(eps)), "q_max": info0["q_max"], "lambda_max": info0["lambda_max"],
              "q_c": info0["q_c"], "sigma_max": info0["sigma_max"],
              "lever_rule_frac_pos": float((rho_mean + np.sqrt(eps)) / (2 * np.sqrt(eps)))}
    # Worst case explicit-Euler bound using the largest f'' seen in any run (rho up to max |rho|).
    rmax = max(np.abs(s["max"]).max() for s in series.values()) if series else 1.0
    rmax = max(rmax, max(np.abs(s["min"]).max() for s in series.values()))
    lam_max = 8 / cfg["dx"] ** 2
    fpp_max = -eps + 3 * rmax**2
    theory["max_|rho|_observed"] = float(rmax)
    theory["dt_stability_limit"] = float(2 / (cfg["M"] * lam_max * (fpp_max + cfg["interface_scalar"] * cfg["k"] * lam_max)))
    theory["dt_used"] = dt

    # ---- dt check
    dtc_dir = out / "dtcheck" / "C_noise_halfdt"
    if dtc_dir.exists() and "C_noise" in runs:
        rc = load_run(dtc_dir, dt / 2)
        sc = rc["rho"].std(axis=(1, 2))
        tm = runs["C_noise"]["t"]
        sm = series["C_noise"]["std"]
        common = tm <= rc["t"][-1]
        s_interp = np.interp(tm[common], rc["t"], sc)
        checks["dtcheck_onset_dt"] = crossing_time(tm, sm, ONSET_STD)
        checks["dtcheck_onset_halfdt"] = crossing_time(rc["t"], sc, ONSET_STD)
        checks["dtcheck_max_rel_RMS_diff"] = float(np.max(np.abs(sm[common] - s_interp) / s_interp))
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.semilogy(tm[common], sm[common], label=f"dt = {dt:g} (study)")
        ax.semilogy(rc["t"], sc, "--", label=f"dt = {dt / 2:g} (check)")
        ax.set_xlabel(T_LABEL), ax.set_ylabel(r"RMS($\rho-\bar\rho$)")
        ax.set_title("Time-step check, profile C (noise)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "dtcheck.png", dpi=DPI)
        plt.close(fig)

    # ---- float64 precision check (same IC and parameters as C_noise, only float_type differs)
    f64_dir = out / "f64check" / "C_noise_f64"
    if f64_dir.exists() and "C_noise" in runs:
        r64 = load_run(f64_dir, dt)
        f64_vals = r64["rho"][-1]
        checks["f64_values_not_float32_representable"] = bool(
            np.any(f64_vals.astype(np.float32).astype(np.float64) != f64_vals))
        m64 = frame_metrics(r64["rho"])
        e64, e32 = r64["energy"], runs["C_noise"]["energy"]
        d64 = m64["mean"] - m64["mean"][0]
        checks["f64_mass_drift_max_abs"] = float(np.abs(d64).max())
        checks["f32_mass_drift_max_abs"] = metrics["C_noise"]["mass_drift_max_abs"]
        checks["f64_onset"] = crossing_time(r64["t"], m64["std"], ONSET_STD)
        checks["f64_F_final"] = float(e64["F"][-1])
        checks["f64_L1_final"] = float(m64["L1"][-1])
        checks["f64_n_domains_pos_final"] = int(m64["n_dom_pos"][-1])
        checks["f64_phase_frac_pos_final"] = float(m64["phase_frac_pos"][-1])
        checks["f64_vs_f32_max_abs_diff_rho_t<=200"] = float(np.abs(
            r64["rho"][r64["t"] <= 200] - runs["C_noise"]["rho"][runs["C_noise"]["t"] <= 200]).max())
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        t32 = runs["C_noise"]["t"]
        axes[0].loglog(t32[1:], np.abs(series["C_noise"]["mean"][1:] - series["C_noise"]["mean"][0]), label="float32")
        axes[0].loglog(r64["t"][1:], np.abs(d64[1:]) + 1e-18, label="float64")
        axes[0].set_ylabel(r"$|\bar\rho(t)-\bar\rho(0)|$"), axes[0].set_title("Mass drift, profile C")
        axes[1].semilogx(e32["t"][1:], e32["F"][1:], label="float32")
        axes[1].semilogx(e64["t"][1:], e64["F"][1:], "--", label="float64")
        axes[1].set_ylabel("F/V"), axes[1].set_title("Free energy, profile C")
        axes[2].loglog(t32[1:], series["C_noise"]["L1"][1:], label="float32")
        axes[2].loglog(r64["t"][1:], m64["L1"][1:], "--", label="float64")
        axes[2].set_ylabel(r"$L$ " + L_LABEL), axes[2].set_title("Characteristic length, profile C")
        for a in axes:
            a.set_xlabel(T_LABEL), a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "precision_check.png", dpi=DPI)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), constrained_layout=True)
        for a, (lab, fr) in zip(axes, (("float32", runs["C_noise"]["rho"][-1]), ("float64", r64["rho"][-1]))):
            im = a.imshow(fr, cmap=CMAP, vmin=-1, vmax=1)
            a.set_title(f"C, {lab}, t = {t32[-1]:g}"), a.set_xticks([]), a.set_yticks([])
        fig.colorbar(im, ax=axes, label=r"$\rho$", shrink=0.8)
        fig.savefig(out / "precision_check_final.png", dpi=DPI)
        plt.close(fig)

    with open(out / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "checks": checks, "theory": theory}, f, indent=2, default=float)

    # ---- metrics table (CSV + figure)
    cols = ["t_onset (RMS>=0.1)", "t_onset linear theory", "t_sat (RMS>=0.9 final)", "t_F50 (half energy drop)",
            "F_final", "L1_final", "k_peak_final", "phase_frac_pos_final", "interface_len_final",
            "n_domains_pos_final", "n_domains_neg_final", "anisotropy_final", "coarsening_exponent",
            "mass_drift_max_abs"]
    with open(out / "metrics_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "profile"] + cols)
        for k in keys:
            w.writerow([k, names[k]] + [metrics[k][c] for c in cols])
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    short_cols = ["t_onset\n(RMS>=0.1)", "t_onset\nlinear\ntheory", "t_sat\n(90% of\nfinal RMS)", "t_F50\n(half F\ndrop)",
                  "F final", "L1 final", "k_peak\nfinal", "area frac\nrho>0", "interface\nlength", "# domains\nrho>0",
                  "# domains\nrho<0", "aniso-\ntropy", "coarsening\nexponent", "max |mass\ndrift|"]
    fig, ax = plt.subplots(figsize=(16, 0.6 * len(keys) + 2.0))
    ax.axis("off")
    tab = ax.table(cellText=[[ic.SHORT_NAMES.get(k, k)] + [fmt(metrics[k][c]) for c in cols] for k in keys],
                   colLabels=["profile"] + short_cols, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 2.6)
    ax.set_title(f"Metrics summary (t_final = {runs[keys[0]]['t'][-1]:g}, N = {N}); lengths in grid units, times in "
                 "simulation units", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "metrics_summary.png", dpi=DPI)
    plt.close(fig)

    # ---- figures
    plot_initial_conditions(ics, names, gc, out)
    T = runs[keys[0]]["t"][-1]
    snap_t = np.concatenate([[0.0], np.geomspace(T / 100, T, 5)])
    plot_snapshots(runs, names, snap_t, out)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c, k in enumerate(keys):
        e = runs[k]["energy"]
        axes[0].plot(np.maximum(e["t"], dt), e["F"], color=f"C{c}", label=names[k])
        axes[1].plot(runs[k]["t"][1:], np.abs(series[k]["mean"][1:] - series[k]["mean"][0]), color=f"C{c}",
                     label=names[k])
    axes[0].set_xscale("log"), axes[0].set_xlabel(T_LABEL), axes[0].set_ylabel("average free energy per bin F/V")
    axes[0].set_title("Free energy vs time (energy.dat)")
    axes[1].set_xscale("log"), axes[1].set_yscale("log"), axes[1].set_xlabel(T_LABEL)
    axes[1].set_ylabel(r"$|\bar\rho(t)-\bar\rho(0)|$ (from trajectory frames)")
    axes[1].set_title("Mass conservation: drift of mean density")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "energy_and_mass_vs_time.png", dpi=DPI)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    T = runs[keys[0]]["t"][-1]
    for c, k in enumerate(keys):
        t = runs[k]["t"]
        axes[0].plot(t, series[k]["std"], color=f"C{c}", label=names[k])
        axes[0].plot(t, lam_theory[k][0], color=f"C{c}", ls=":", lw=1)
        axes[1].plot(t, series[k]["max"], color=f"C{c}", label=names[k])
        axes[1].plot(t, series[k]["min"], color=f"C{c}", ls="--")
    axes[0].axhline(ONSET_STD, color="k", lw=0.7, ls="--")
    axes[0].set_xscale("symlog", linthresh=10), axes[0].set_yscale("log")
    for a in axes:
        a.set_xlim(0, T * 1.1)
    axes[0].set_ylim(1e-3, 2)
    axes[0].set_xlabel(T_LABEL), axes[0].set_ylabel(r"RMS($\rho-\bar\rho$) = $\sqrt{\mathrm{Var}\,\rho}$")
    axes[0].set_title("Order-parameter RMS (solid) and linear theory (dotted)\ndashed: onset threshold 0.1")
    axes[1].axhline(np.sqrt(eps), color="gray", lw=0.6), axes[1].axhline(-np.sqrt(eps), color="gray", lw=0.6)
    axes[1].set_xscale("symlog", linthresh=10)
    axes[1].set_xlabel(T_LABEL), axes[1].set_ylabel(r"max $\rho$ (solid), min $\rho$ (dashed)")
    axes[1].set_title(r"Extrema of $\rho$; gray: binodal $\pm\sqrt{\epsilon}$")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "variance_vs_time.png", dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for c, k in enumerate(keys):
        t, L = runs[k]["t"], series[k]["L1"]
        ax.loglog(t[1:], L[1:], color=f"C{c}", label=f"{names[k]}  (n = {metrics[k]['coarsening_exponent']:.2f})")
        tf0, tf1 = metrics[k]["fit_t_range"]
        n_fit = metrics[k]["coarsening_exponent"]
        if np.isfinite(n_fit):
            sel = (t >= tf0) & (t <= tf1)
            Lfit = np.exp(np.polyfit(np.log(t[sel]), np.log(L[sel]), 1)[1]) * t[sel] ** n_fit
            ax.loglog(t[sel], Lfit, color=f"C{c}", ls=":", lw=2)
    tt = np.geomspace(T / 20, T, 10)
    ax.loglog(tt, 0.9 * series[keys[0]]["L1"][-1] * (tt / T) ** (1 / 3), "k--", lw=1, label=r"$t^{1/3}$ reference")
    ax.axhline(theory["lambda_max"], color="gray", lw=0.7, ls="-.", label=r"$\lambda_{max}$ linear theory")
    ax.set_xlabel(T_LABEL), ax.set_ylabel(r"$L = 2\pi/\langle k\rangle$ " + L_LABEL)
    ax.set_title("Characteristic length vs time (dotted: power-law fits)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "length_vs_time.png", dpi=DPI)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c, k in enumerate(keys):
        ks, Sr = radial_average(structure_factor(runs[k]["rho"][-1:]))
        axes[0].semilogy(ks, Sr[0], "o-", ms=3, color=f"C{c}", label=names[k])
        ks0, Sr0 = radial_average(structure_factor(ics[k]))
        axes[1].semilogy(ks0, Sr0[0] + 1e-30, "o-", ms=3, color=f"C{c}", label=names[k])
    for a in axes:
        a.axvline(theory["q_max"], color="gray", ls="-.", lw=0.8, label=r"$q_{max}$")
        a.axvline(theory["q_c"], color="gray", ls=":", lw=0.8, label=r"$q_c$ (marginal)")
        a.set_xlabel(r"|k| [rad / grid unit]")
    axes[0].set_ylabel("radially averaged S(k)")
    axes[0].set_title(f"Final structure factor, t = {T:g}")
    axes[1].set_ylim(1e-12, None)
    axes[1].set_title("Initial structure factor (t = 0)")
    axes[1].set_ylabel("radially averaged S(k)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "structure_factor.png", dpi=DPI)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for c, k in enumerate(keys):
        t = runs[k]["t"]
        axes[0].plot(t[1:], series[k]["interface_len"][1:], color=f"C{c}", label=names[k])
        axes[1].plot(t[1:], series[k]["n_dom_pos"][1:], color=f"C{c}")
        axes[2].plot(t[1:], series[k]["phase_frac_pos"][1:], color=f"C{c}")
    axes[2].axhline(theory["lever_rule_frac_pos"], color="k", ls="--", lw=0.8, label="lever rule")
    for a in axes:
        a.set_xscale("log"), a.set_xlabel(T_LABEL)
    axes[0].set_ylabel(r"interface length " + L_LABEL), axes[0].set_title(r"Interface length ($\rho = 0$ contour)")
    axes[1].set_ylabel(r"# domains with $\rho > 0$"), axes[1].set_title("Domain count (periodic, 4-connected)")
    axes[2].set_ylabel(r"area fraction $\rho > 0$"), axes[2].set_title("Phase fraction")
    axes[0].legend(fontsize=7), axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "morphology_vs_time.png", dpi=DPI)
    plt.close(fig)

    # linear theory dispersion
    fig, ax = plt.subplots(figsize=(6, 4))
    q = np.linspace(0, 1.0, 300)
    c_ = cfg["interface_scalar"] * cfg["k"]
    ax.plot(q, -q**2 * (theory["f''(rho_mean)"] + c_ * q**2), "k", label="continuum")
    lam = info0["lam"]
    qd = np.sqrt(lam).ravel()
    ax.plot(qd, info0["sigma"].ravel(), ".", ms=2, color="C3", label=r"lattice modes, plotted at $q_{eff}=\sqrt{\lambda}$ ($\lambda$: eigenvalue of $-\nabla^2_h$)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_ylim(-0.3, 0.1), ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"$|q|$ [rad / grid unit]"), ax.set_ylabel(r"growth rate $\sigma(q)$ [1 / time unit]")
    ax.set_title(r"Linear stability at $\bar\rho$ = %.2f: $\sigma = -Mq^2(f''+2kq^2)$" % rho_mean)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "linear_dispersion.png", dpi=DPI)
    plt.close(fig)

    if not args.no_gif:
        for k in keys:
            write_gif(runs[k], names[k], out / f"evolution_{k}.gif")

    print(json.dumps({"checks": checks, "theory": theory}, indent=1, default=float))
    for k in keys:
        print(k, {c: (round(metrics[k][c], 4) if isinstance(metrics[k][c], float) else metrics[k][c]) for c in cols})


if __name__ == "__main__":
    main()
