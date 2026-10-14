"""Build per-profile configs and initial conditions, then run every simulation through pfm.run.

Examples (run from any directory):
    python run_study.py --N 32 --steps 4000 --traj-every 200 --print-every 20 --out results_smoke --config-dir results_smoke/configs
    python run_study.py                      # full study with configs/base.toml, outputs in results/
    python run_study.py --analyze            # also run analysis.py afterwards
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import toml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import initial_conditions as ic  # noqa: E402

PFM_KEYS_VARYING = {"load_from", "write_path"}


def rel(p: Path) -> str:
    """Path relative to the study directory (the cwd of every simulation), with forward slashes."""
    return Path(os.path.relpath(p.resolve(), HERE)).as_posix()


def build(args: argparse.Namespace) -> tuple[list[Path], dict]:
    """Write ICs and per-profile TOMLs; return the TOML paths and the study manifest."""
    base = toml.load(args.base)
    study = base.pop("study")
    for key, val in (("N", args.N), ("steps", args.steps), ("print_trajectory_every", args.traj_every),
                     ("print_every", args.print_every)):
        if val is not None:
            base[key] = val
    N, seed = int(base["N"]), int(base["seed"])
    rho_mean, amp = float(study["rho_mean"]), float(study["amp_rms"])

    out, cfg_dir = Path(args.out).resolve(), Path(args.config_dir).resolve()
    (out / "ics").mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    profiles = ic.make_all_profiles(N, rho_mean, amp, seed)
    keys = args.profiles or list(profiles)
    for key, rho in profiles.items():
        ic.write_load_from(out / "ics" / f"{key}.dat", rho)
        np.save(out / "ics" / f"{key}.npy", rho)
    tomls, cfgs = [], []
    for key in keys:
        ic_path = out / "ics" / f"{key}.dat"
        cfg = copy.deepcopy(base)
        cfg["load_from"] = rel(ic_path)
        cfg["write_path"] = rel(out / "runs" / key)
        path = cfg_dir / f"{key}.toml"
        with open(path, "w") as f:
            toml.dump(cfg, f)
        tomls.append(path)
        cfgs.append(cfg)

    # Frozen-parameter check: every generated config is identical apart from the two path keys.
    strip = [{k: v for k, v in c.items() if k not in PFM_KEYS_VARYING} for c in cfgs]
    assert all(s == strip[0] for s in strip), "Configs differ in more than load_from/write_path"

    if args.dt_check_time:
        cfg = copy.deepcopy(base)
        cfg["dt"] = base["dt"] / 2
        cfg["steps"] = int(round(args.dt_check_time / cfg["dt"]))
        cfg["print_every"] = base["print_every"] * 2
        cfg["print_trajectory_every"] = base["print_trajectory_every"] * 2
        cfg["load_from"] = rel(out / "ics" / "C_noise.dat")
        cfg["write_path"] = rel(out / "dtcheck" / "C_noise_halfdt")
        path = cfg_dir / "dtcheck_C_noise_halfdt.toml"
        with open(path, "w") as f:
            toml.dump(cfg, f)
        tomls.append(path)

    if args.f64_check:
        cfg = copy.deepcopy(base)
        cfg["float_type"] = "float64"
        cfg["load_from"] = rel(out / "ics" / "C_noise.dat")
        cfg["write_path"] = rel(out / "f64check" / "C_noise_f64")
        path = cfg_dir / "f64check_C_noise.toml"
        with open(path, "w") as f:
            toml.dump(cfg, f)
        tomls.append(path)

    manifest = {"base_config": rel(Path(args.base)), "frozen": strip[0], "study": study,
                "profiles": keys, "names": {k: ic.PROFILE_NAMES[k] for k in keys},
                "gauss_center_ij": list(ic.gauss_center(N)), "configs": [rel(p) for p in tomls]}
    return tomls, manifest


def run_one(toml_path: Path) -> float:
    """Run one config in a fresh interpreter (cwd = study dir) and return wall time in seconds."""
    t0 = time.time()
    log = toml_path.with_suffix(".log")
    with open(log, "w") as f:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--run-config", str(toml_path)],
                              cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Simulation failed for {toml_path}; see {log}")
    return time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=str(HERE / "configs" / "base.toml"))
    p.add_argument("--out", default=str(HERE / "results"), help="output directory")
    p.add_argument("--config-dir", default=str(HERE / "configs"), help="where per-profile TOMLs are written")
    p.add_argument("--N", type=int, default=None, help="grid size override (power of 2)")
    p.add_argument("--steps", type=int, default=None, help="number of steps override")
    p.add_argument("--traj-every", type=int, default=None, help="print_trajectory_every override")
    p.add_argument("--print-every", type=int, default=None, help="print_every override")
    p.add_argument("--profiles", nargs="*", default=None, choices=list(ic.RAW_GENERATORS))
    p.add_argument("--jobs", type=int, default=3, help="simulations run in parallel")
    p.add_argument("--dt-check-time", type=float, default=0.0,
                   help="if > 0, also run C_noise with dt/2 up to this time (numerical check only)")
    p.add_argument("--f64-check", action="store_true",
                   help="also run C_noise in float64 (precision diagnostic only)")
    p.add_argument("--only-checks", action="store_true",
                   help="run only the dt/f64 diagnostic configs (profiles already run); keeps manifest wall times")
    p.add_argument("--analyze", action="store_true", help="run analysis.py on --out afterwards")
    p.add_argument("--run-config", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.run_config:
        import pfm
        minutes, seconds = pfm.run(args.run_config)
        print(f"finished {args.run_config} in {minutes} min {seconds} s")
        return

    tomls, manifest = build(args)
    old_manifest = Path(args.out) / "manifest.json"
    wall = json.loads(old_manifest.read_text()).get("wall_time_s", {}) if old_manifest.exists() else {}
    if args.only_checks:
        tomls = [t for t in tomls if t.stem.startswith(("dtcheck", "f64check"))]
    print(f"Running {len(tomls)} configs from {args.config_dir} with {args.jobs} parallel jobs")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        times = list(pool.map(run_one, tomls))
    manifest["wall_time_s"] = wall | {t.stem: round(s, 1) for t, s in zip(tomls, times)}
    for t, s in zip(tomls, times):
        print(f"  {t.stem}: {s:.1f} s")
    with open(Path(args.out) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    if args.analyze:
        subprocess.run([sys.executable, str(HERE / "analysis.py"), "--out", str(Path(args.out).resolve())],
                       check=True)


if __name__ == "__main__":
    main()
