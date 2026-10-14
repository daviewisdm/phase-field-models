"""Initial-condition generators and file I/O for the initial-condition study.

Array layout (confirmed in REPO_NOTES.md and by roundtrip_test.py):
    rho has shape (Ns, N, N) with Ns = 1 for Landau. rho[0, i, j]: i is the row index
    (axis 1, one text line per i in load_from / trajectory files), j is the column index
    (axis 2, whitespace-separated values within a line).

Fairness definition used for every profile:
    mean(rho) = rho_mean exactly, and RMS(rho - mean(rho)) = sqrt(mean((rho - rho_mean)**2)) = amp_rms.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np

# Profile key -> human readable name, used for files and plot labels.
PROFILE_NAMES: dict[str, str] = {
    "A_stripe1": "A: single-mode stripe (N_peaks=1)",
    "B_multimode": "B: 2D product of cosines (4x4)",
    "C_noise": "C: uncorrelated Gaussian noise",
    "D_gaussian": "D: off-center Gaussian bump",
    "E_ring": "E: concentric rings (target)",
}

SHORT_NAMES: dict[str, str] = {
    "A_stripe1": "A: stripe, n=1",
    "B_multimode": "B: cos x cos, 4x4",
    "C_noise": "C: white noise",
    "D_gaussian": "D: Gaussian bump",
    "E_ring": "E: rings",
}

# Location of the Gaussian bump in (row i, column j) as a fraction of N; deliberately asymmetric.
GAUSS_CENTER_FRAC: tuple[float, float] = (0.25, 0.65)


def index_grid(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (i, j) integer index grids of shape (N, N) with i = row (axis 1 of rho), j = column."""
    return np.meshgrid(np.arange(N), np.arange(N), indexing="ij")


def periodic_distance(N: int, ci: float, cj: float) -> np.ndarray:
    """Minimum-image distance from (ci, cj) on an N x N periodic grid with dx = 1."""
    i, j = index_grid(N)
    di = np.abs(i - ci)
    dj = np.abs(j - cj)
    di = np.minimum(di, N - di)
    dj = np.minimum(dj, N - dj)
    return np.sqrt(di**2 + dj**2)


def normalize(raw: np.ndarray, rho_mean: float, amp_rms: float) -> np.ndarray:
    """Shift and scale a raw pattern so it has mean rho_mean and RMS deviation amp_rms."""
    delta = raw - raw.mean()
    rms = np.sqrt(np.mean(delta**2))
    if rms == 0:
        raise ValueError("Raw pattern is constant; cannot normalize amplitude.")
    return rho_mean + delta * (amp_rms / rms)


def raw_stripe1(N: int, seed: int) -> np.ndarray:
    """Single Fourier mode along the row axis i: cos(2*pi*i/N)."""
    i, _ = index_grid(N)
    return np.cos(2 * np.pi * 1 * i / N)


def raw_multimode(N: int, seed: int, n_peaks: int = 4) -> np.ndarray:
    """2D product of cosines with n_peaks periods along each axis."""
    i, j = index_grid(N)
    return np.cos(2 * np.pi * n_peaks * i / N) * np.cos(2 * np.pi * n_peaks * j / N)


def raw_noise(N: int, seed: int) -> np.ndarray:
    """Spatially uncorrelated standard normal noise from numpy's default_rng(seed)."""
    return np.random.default_rng(seed).standard_normal((N, N))


def raw_gaussian(N: int, seed: int) -> np.ndarray:
    """Single Gaussian bump (sigma = N/16) at GAUSS_CENTER_FRAC, periodic distance."""
    ci, cj = gauss_center(N)
    r = periodic_distance(N, ci, cj)
    sigma = N / 16
    return np.exp(-(r**2) / (2 * sigma**2))


def raw_ring(N: int, seed: int) -> np.ndarray:
    """Concentric rings cos(2*pi*r/(N/4)) around the domain center, periodic distance."""
    r = periodic_distance(N, N / 2, N / 2)
    return np.cos(2 * np.pi * r / (N / 4))


def gauss_center(N: int) -> tuple[int, int]:
    """Integer (row, column) location of the Gaussian bump peak."""
    return int(round(GAUSS_CENTER_FRAC[0] * N)), int(round(GAUSS_CENTER_FRAC[1] * N))


RAW_GENERATORS: dict[str, Callable[[int, int], np.ndarray]] = {
    "A_stripe1": raw_stripe1,
    "B_multimode": raw_multimode,
    "C_noise": raw_noise,
    "D_gaussian": raw_gaussian,
    "E_ring": raw_ring,
}


def make_profile(key: str, N: int, rho_mean: float, amp_rms: float, seed: int) -> np.ndarray:
    """Build profile `key` as a float64 array of shape (1, N, N) in the repo's rho layout."""
    rho = normalize(RAW_GENERATORS[key](N, seed), rho_mean, amp_rms)
    return rho[None, :, :]


def make_all_profiles(N: int, rho_mean: float, amp_rms: float, seed: int) -> dict[str, np.ndarray]:
    """Build every profile and assert the fair-comparison constraints."""
    profiles = {k: make_profile(k, N, rho_mean, amp_rms, seed) for k in RAW_GENERATORS}
    check_fairness(profiles, rho_mean, amp_rms)
    return profiles


def check_fairness(profiles: dict[str, np.ndarray], rho_mean: float, amp_rms: float,
                   tol: float = 1e-12) -> None:
    """Assert identical shape, mean and RMS amplitude, both in float64 and after float32 rounding."""
    shapes = {p.shape for p in profiles.values()}
    assert len(shapes) == 1, f"Profiles have different shapes: {shapes}"
    for key, rho in profiles.items():
        m = rho.mean()
        a = np.sqrt(np.mean((rho - m) ** 2))
        assert abs(m - rho_mean) < tol, f"{key}: mean {m} != {rho_mean}"
        assert abs(a - amp_rms) < tol, f"{key}: RMS {a} != {amp_rms}"
        r32 = rho.astype(np.float32).astype(np.float64)
        assert abs(r32.mean() - rho_mean) < 1e-6, f"{key}: float32 mean drift {r32.mean() - rho_mean}"
        assert abs(np.sqrt(np.mean((r32 - r32.mean()) ** 2)) - amp_rms) < 1e-6 * max(amp_rms, 1.0)


def write_load_from(path: Path | str, rho: np.ndarray) -> None:
    """Write rho (shape (1, N, N)) in the format read by PhaseFieldModel's `load_from` branch.

    One '#' header line (skipped by the loader), then N lines of N values; line i holds rho[0, i, :].
    Values are float32-rounded and written with 9 significant digits, which round-trips float32 exactly.
    """
    assert rho.ndim == 3 and rho.shape[0] == 1 and rho.shape[1] == rho.shape[2], rho.shape
    N = rho.shape[1]
    r32 = rho[0].astype(np.float32)
    with open(path, "w") as f:
        f.write(f"# step = 0, t = 0.00000, size = {N}x{N}\n")
        for row in r32:
            f.write(" ".join(f"{v:.9g}" for v in row) + "\n")
        f.write("\n")


_HEADER_RE = re.compile(r"#\s*step\s*=\s*(\d+)(?:,\s*t\s*=\s*([-\d.eE+]+))?.*size\s*=\s*([\dx]+)")


def read_frames(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a trajectory / init_ / last_ file written by PhaseFieldModel.print_species_density.

    Each frame: '# step = S, t = T, size = NxN' then N lines of N values, then a blank line.
    Returns (steps, times, frames) with frames of shape (F, N, N) as float64; frames[f, i, j] = rho[0, i, j].
    `times` is NaN if the header carries no time field.
    """
    steps: list[int] = []
    times: list[float] = []
    frames: list[np.ndarray] = []
    rows: list[list[float]] = []
    n_expected = 0

    def flush() -> None:
        if steps:
            arr = np.array(rows, dtype=np.float64)
            assert arr.shape == (n_expected, n_expected), f"Bad frame shape {arr.shape} at step {steps[-1]}"
            frames.append(arr)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                flush()
                m = _HEADER_RE.match(line)
                assert m, f"Unrecognized header: {line}"
                steps.append(int(m.group(1)))
                times.append(float(m.group(2)) if m.group(2) else np.nan)
                dims = [int(d) for d in m.group(3).split("x")]
                assert len(dims) == 2 and dims[0] == dims[1], f"Only square 2D supported: {dims}"
                n_expected = dims[0]
                rows = []
            else:
                rows.append([float(v) for v in line.split()])
    flush()
    return np.array(steps), np.array(times), np.stack(frames)


def read_energy(path: Path | str) -> dict[str, np.ndarray]:
    """Parse energy.dat (CPU path): columns 't F_avg mass_avg step', formats %.5f %.8f %.5f %d."""
    data = np.loadtxt(path, ndmin=2)
    assert data.shape[1] == 4, f"energy.dat has {data.shape[1]} columns, expected 4"
    return {"t": data[:, 0], "F": data[:, 1], "mass": data[:, 2], "step": data[:, 3].astype(int)}
