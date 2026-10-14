"""Step 0 round-trip test: confirm the load_from format, orientation, and output parsing against the real code.

Run from anywhere:  python roundtrip_test.py
Writes scratch files to results/roundtrip/ and exits non-zero if any check fails.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import toml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import initial_conditions as ic  # noqa: E402

OUT = HERE / "results" / "roundtrip"


def base_cfg(N: int) -> dict:
    return {
        "steps": 2, "print_every": 1, "print_trajectory_every": 1, "N": N, "dt": 0.001,
        "free_energy": "landau", "integrator": "euler", "model": "ch", "dim": 2,
        "interface_scalar": 2.0, "k": 1.0, "M": 1.0, "float_type": "float32", "seed": 7,
        "verbose": False, "landau": {"epsilon": 0.9},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT)
    import jax
    import jax.numpy as jnp
    from pfm import SimulationManager
    N = 32

    # 1) Default initial condition (initial_density branch) and an independent re-implementation of its formula.
    cfg = base_cfg(N) | {"initial_density": 0.01, "initial_A": 0.001, "initial_N_peaks": 1, "write_path": "default"}
    default_rho = np.array(SimulationManager(cfg).init_field)
    x, y = jnp.meshgrid(jnp.arange(N), jnp.arange(N))
    r = jnp.sqrt(x**2 + y**2)
    mod = 0.001 * jnp.cos(2 * np.pi * 1 / N * r)
    _, sub = jax.random.split(jax.random.PRNGKey(7))
    noise = jax.random.uniform(sub, shape=(1, N, N))
    mine = 0.01 * (1.0 + 2.0 * mod[None] * (1.0 + 0.02 * (noise - 0.5)))
    assert default_rho.shape == (1, N, N), default_rho.shape
    assert default_rho.dtype == np.float32, default_rho.dtype
    assert np.allclose(default_rho, np.array(mine, dtype=np.float32), rtol=0, atol=1e-9), "Default IC formula mismatch"
    print(f"[ok] default IC: shape {default_rho.shape}, dtype {default_rho.dtype}, formula reproduced; "
          f"max at (i,j)={np.unravel_index(default_rho[0].argmax(), (N, N))} (corner-centred rings)")

    # 2) Write default IC in load_from format and read it back through the repo's loader.
    ic.write_load_from(OUT / "default_ic.dat", default_rho.astype(np.float64))
    cfg_l = base_cfg(N) | {"load_from": "default_ic.dat", "write_path": "load_default"}
    back = np.array(SimulationManager(cfg_l).init_field)
    assert back.shape == default_rho.shape and np.array_equal(back, default_rho), "default IC round-trip mismatch"
    print("[ok] default IC round-trip through load_from is bit-exact")

    # 3) Asymmetric field: off-centre Gaussian plus a ramp along rows only. A transpose would be detected.
    asym = ic.make_profile("D_gaussian", N, -0.2, 0.01, 0)
    asym[0] += 1e-3 * np.arange(N)[:, None]
    ic.write_load_from(OUT / "asym_ic.dat", asym)
    cfg_a = base_cfg(N) | {"load_from": "asym_ic.dat", "write_path": "load_asym"}
    back_a = np.array(SimulationManager(cfg_a).init_field)
    asym32 = asym.astype(np.float32)
    assert np.array_equal(back_a, asym32), "asymmetric round-trip mismatch"
    assert not np.allclose(back_a[0], asym32[0].T), "test field is not asymmetric enough"
    peak = np.unravel_index((back_a[0] - 1e-3 * np.arange(N)[:, None]).argmax(), (N, N))
    print(f"[ok] asymmetric round-trip bit-exact; Gaussian peak at (i,j)={tuple(int(p) for p in peak)}, "
          f"intended {ic.gauss_center(N)}; file line i == rho[0, i, :]")
    assert tuple(int(p) for p in peak) == ic.gauss_center(N)

    # 4) Custom initial condition path (MagneticFilm style) gives the same array.
    cfg_c = base_cfg(N) | {"custom_initial_condition": True, "write_path": "custom"}

    def fn(z: np.ndarray) -> np.ndarray:
        z[:] = asym
        return z

    back_c = np.array(SimulationManager(cfg_c, custom_initial_condition=fn).init_field)
    assert np.array_equal(back_c, asym32), "custom_initial_condition path mismatch"
    print("[ok] custom_initial_condition path gives identical array")

    # 5) Real 2-step run via pfm.main.run with load_from; parse trajectory, init_, last_, energy files.
    cfg_r = base_cfg(N) | {"load_from": "asym_ic.dat", "write_path": "run_asym"}
    with open("run_asym.toml", "w") as f:
        toml.dump(cfg_r, f)
    import pfm
    pfm.run("run_asym.toml")
    files = sorted(p.name for p in Path("run_asym").iterdir())
    print(f"[info] files written by CPU path: {files}")
    steps, times, frames = ic.read_frames("run_asym/trajectory_0.dat")
    print(f"[info] trajectory steps {steps.tolist()}, times {times.tolist()}, frames {frames.shape}")
    assert np.array_equal(frames[0].astype(np.float32), asym32[0]), "trajectory frame 0 != IC"
    _, _, init_f = ic.read_frames("run_asym/init_0.dat")
    assert np.array_equal(init_f[0].astype(np.float32), asym32[0]), "init_0.dat != IC"
    _, _, last_f = ic.read_frames("run_asym/last_0.dat")
    e = ic.read_energy("run_asym/energy.dat")
    print(f"[info] energy.dat rows: t={e['t'].tolist()} F={e['F'].tolist()} mass={e['mass'].tolist()} step={e['step'].tolist()}")
    print(f"[ok] trajectory frame 0 and init_0.dat equal the IC exactly; last_0.dat shape {last_f.shape}")

    # 6) One explicit Euler step reproduced by hand from the update in explicit_euler.py.
    def lap(p: np.ndarray) -> np.ndarray:
        return -4 * p + np.roll(p, 1, 0) + np.roll(p, -1, 0) + np.roll(p, 1, 1) + np.roll(p, -1, 1)

    r0 = asym32[0].astype(np.float64)
    mu = (-0.9 * r0 + r0**3) - 2.0 * 1.0 * lap(r0)
    r1 = r0 + 1.0 * lap(mu) * 0.001
    err = np.abs(frames[1] - r1).max()
    print(f"[ok] frame 1 matches hand-computed Euler step, max |diff| = {err:.2e} (float32 rounding)")
    assert err < 1e-6
    print("ALL ROUND-TRIP CHECKS PASSED")


if __name__ == "__main__":
    main()
