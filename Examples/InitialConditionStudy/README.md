# Initial-condition study (Cahn-Hilliard, Landau free energy)

A controlled study of how the spatial pattern of the initial perturbation changes the evolution of a 2D
Cahn-Hilliard / Landau system. Five profiles with identical mean density and identical RMS amplitude are run with
otherwise frozen parameters. Results are in [REPORT.md](REPORT.md); repo findings that the code relies on (file
formats, array layout, output parsing) are in [REPO_NOTES.md](REPO_NOTES.md).

## Files

| File | Purpose |
|------|---------|
| `REPO_NOTES.md` | Step 0 code investigation with file / line citations |
| `roundtrip_test.py` | Verifies `load_from` layout, default IC formula, custom-IC path and trajectory parsing on real runs |
| `initial_conditions.py` | Profile generators, mean / RMS normalisation and assertions, `load_from` writer, output parsers |
| `configs/base.toml` | The single base config; `configs/<profile>.toml` are generated from it |
| `run_study.py` | Builds ICs and configs, runs every simulation via `pfm.run` in a fresh subprocess |
| `analysis.py` | Metrics, figures, `metrics_summary.csv`, `metrics.json` |
| `REPORT.md` | Short report |

## Environment

Python 3.12 (conda env), `pip install -e .` from the repo root, plus `scipy` (domain labelling) and `pillow`
(GIFs). Versions used: jax / jaxlib 0.11.2 (CPU only, no GPU detected), numpy 2.2.4, matplotlib 3.11.2,
scipy 1.18.1, toml 0.10.2. Windows 11, 4-core CPU.

```
conda create -n pfm-icstudy python=3.12
conda activate pfm-icstudy
pip install -e .            # from the repo root
pip install scipy pillow
```

## Reproduce

All commands are run from this folder. Each simulation runs with this folder as its working directory, so the
`load_from` / `write_path` entries in the TOMLs are relative to it.

```
python roundtrip_test.py                                   # Step 0 checks (about 1 min)
python run_study.py --N 32 --steps 20000 --traj-every 200 --print-every 50 \
       --out results_smoke --config-dir results_smoke/configs --dt-check-time 50 --analyze   # smoke test (about 2 min)
python run_study.py --dt-check-time 300 --f64-check --jobs 3   # full study + diagnostics
python analysis.py --out results                            # metrics and figures (about 2 min)
```

`run_study.py` options: `--N`, `--steps`, `--traj-every`, `--print-every` (override the base config), `--out`,
`--config-dir`, `--profiles`, `--jobs`, `--dt-check-time T` (also run profile C with dt/2 up to time T),
`--f64-check` (also run profile C in float64), `--only-checks` (run only the diagnostics), `--analyze`.

Everything is deterministic: the only randomness is the noise profile C (`numpy.random.default_rng(seed)` with the
config `seed`), and the explicit Euler update itself is deterministic.

## Runtime (measured)

Full study, N = 64, 1,000,000 steps (t = 5000) per profile, 3 jobs in parallel on 4 cores: about 6 min per
profile and 12 min wall time for all five. Run alone (2 jobs), the half-dt check (t = 300) took 37 s and the float64
run of profile C took 204 s. Analysis takes about 1.5 min. On this machine the CPU code path (`_run_cpu`) is used;
on a GPU the repo switches to `_run_jax`, which writes `trajectory_species_0.dat` instead (see REPO_NOTES.md), and
`analysis.load_run` would need that file name.

## Outputs (`results/`)

- `initial_conditions.png`, `snapshots_<profile>.png`, `snapshots_all.png`, `evolution_<profile>.gif`
- `energy_and_mass_vs_time.png`, `variance_vs_time.png`, `length_vs_time.png`, `structure_factor.png`,
  `morphology_vs_time.png`, `linear_dispersion.png`, `dtcheck.png`, `precision_check*.png`
- `metrics_summary.csv` / `.png`, `metrics.json` (all metrics, checks, theory values), `timeseries_<profile>.csv`
- `ics/` (initial conditions as `load_from` text and `.npy`), `manifest.json` (frozen parameters, wall times)
- `runs/`, `dtcheck/`, `f64check/`: raw repo output (about 77 MB of text per run; git-ignored)
