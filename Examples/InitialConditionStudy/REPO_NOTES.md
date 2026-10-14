# Repo notes (Step 0 findings)

Repo state: commit `c909783`. All line numbers refer to that commit. Every statement below was read from source and
confirmed by `roundtrip_test.py` (all checks pass; output reproduced at the bottom).

Note on the instructions: the README and the task refer to `Examples/Landau/run.py`. The folder is actually
`Examples/landau/` (lowercase) and contains only TOML files and `standard_input.dat`; there is no `run.py`. The
TOMLs `euler_noautodiff.toml` / `euler_autodiff.toml` were used as the baseline (`N = 64`, `dt = 0.001`,
`interface_scalar = 2.0`, `[landau] epsilon = 0.9`, `initial_density = 0.01`, `float64`).

## 1. `load_from` (initial-condition file)

Handled inline in `PhaseFieldModel.__init__`, `pfm/phase_field_models/phase_field_model.py` lines 33-77.

- **Precedence** (lines 33, 78, 129, 135-136): `load_from` > `initial_density` > `custom_initial_condition`. If
  `load_from` is present, the other keys are ignored. `ValueError` if none is present.
- **File type**: plain ASCII text, opened with `open(filename, 'r')` (line 35). Path is relative to the current
  working directory of the Python process.
- **Header**: the first line is read with `readline()` (line 36). If it starts with `#` it is skipped (37-38).
  If it does not, the 2D branch (lines 47-54) tries to assign `self.dim = 2` values to a row of length N, which
  fails for N != 2. So **a `#` header line is effectively mandatory in 2D**.
- **Layout (2D)**: lines 59-74. After the header, `for ct, line in enumerate(load_from)`: each non-empty line is
  split on whitespace; it must contain exactly `N` values (else `Exception('Invalid load_from file')`, line 74), and
  it is stored as `initial_field[s, ct, :]`. So **line `i` after the header is `rho[0, i, :]`**: file line = axis 1
  (row index `i`), column within the line = axis 2 (`j`). No transpose.
- `ct` counts every line after the header, including blank lines, so a blank line *between* data lines would shift
  rows; trailing blank lines are harmless (skipped by `if line:`).
- **Species**: the `for s in range(num_species)` loop (line 59) exhausts the file during `s = 0`, so only one species
  is ever read. Fine for Landau (`N_species() == 1`, `pfm/energy_models/landau.py` line 18).
- **Validation**: only the per-line value count. No check on the number of rows: fewer than N rows silently leaves
  zeros; more than N rows raises `IndexError`. No check of `dim` or species count against the file. The `size = NxN`
  header text is ignored.
- **Values / dtype**: parsed with `float()` into `np.zeros((Ns, N, N))` (float64, line 25), then divided by
  `inverse_scaling_factor**3` (line 139; equals 1 since Landau forbids `distance_scaling_factor != 1`, `landau.py`
  lines 14-15), then cast with `jnp.array(initial_field, dtype=config['float_type'])` (line 145) and stored as
  `self.init_rho`. Values are **densities** (the order parameter rho itself), not deviations.
- The repo's own `Examples/landau/standard_input.dat` matches this: 66 lines = 1 `#` header + 64 rows x 64 values +
  1 trailing blank line, i.e. exactly the trajectory frame format below.

## 2. Default initial condition (`initial_density`, `initial_A`, `initial_N_peaks`)

`phase_field_model.py` lines 78-109 (2D: 97-109). With `rho_0 = initial_density`, `A = initial_A` (default `1e-2`,
line 81; README says 0.001), `n = initial_N_peaks` (default 0), `k = 2*pi*n/N` (line 83):

```
x, y = jnp.meshgrid(arange(N), arange(N))          # default 'xy' indexing: x[i, j] = j, y[i, j] = i
r = sqrt(x**2 + y**2)                              # distance from the CORNER (0, 0), not periodic
modulation = A * cos(k * r)
u = jax.random.uniform(split(PRNGKey(seed))[1], (Ns, N, N))
random_factor = u - 0.5            if n == 0
              = 1 + 0.02*(u - 0.5) otherwise
rho = rho_0 * (1 + 2 * modulation * random_factor)
```

Consequences: (a) in 2D `initial_N_peaks` produces quarter-circle rings centred on the corner, not stripes, and the
pattern is not periodic across the boundary; (b) the perturbation is multiplicative in `rho_0`, so `rho_0 = 0` gives
a perfectly flat field; (c) with `n = 0` it is uniform noise of relative amplitude `A`. `roundtrip_test.py`
re-implements this formula independently and reproduces `init_rho` to 1e-9. Because of (a) and (b), this study builds
its own profiles and passes them via `load_from`.

`custom_initial_condition` (lines 129-133): requires the key `custom_initial_condition` in the config and neither of the
other two keys; the Python callable receives `np.zeros((Ns, N, N))` and must return the same shape. Verified to give
the identical array as `load_from` in `roundtrip_test.py`.

## 3. Output files

The manager picks the writer in `SimulationManager.run` (`pfm/manager.py` lines 231-248): `_run_jax` if a GPU/TPU is
found or `override_use_jax=True`, else `_run_cpu`. **This machine has CPU only (`jax.devices() == [CpuDevice(id=0)]`),
so all runs here use `_run_cpu` (lines 128-172).** The two paths write different file names and formats.

### CPU path (used here)

- `energy.dat` (lines 146-165): one line every `print_every` steps for `t in range(steps)` (step 0 included, final
  step `steps` NOT included). Format `f"{t*dt:.5f} {F_avg:.8f} {mass_avg:.5f} {t}"`: time, average free energy per
  bin, average mass (only **5 decimals**, too coarse for a drift check; drift is measured from trajectory frames
  instead), step. `F_avg` is `CahnHilliard.average_free_energy` (`cahn_hilliard.py` lines 22-41):
  `mean over bins of [ -eps/2 rho^2 + rho^4/4 + k * |grad rho|^2 ]` with forward differences (`phase_field_model.py`
  lines 159-175). `mass_avg` is `average_mass` (lines 177-184), the spatial mean of rho.
- `trajectory_0.dat` (lines 137-140, 153-157): a frame every `print_trajectory_every` steps for `t in range(steps)`,
  so the last frame is at the largest multiple below `steps`. Each frame is written by `print_species_density`
  (`phase_field_model.py` lines 187-202): header `# step = {t}, t = {t*dt:.5f}, size = NxN`, then N lines of N values
  (line `i` = `rho[0, i, :]`), then one blank line. Values are written with `f"{val}"` (lines 194, 197) on an
  `np.float32` scalar; the f-string goes through Python `float`, so each value is the exact float64 repr of the
  float32 number (e.g. `0.10000000149011612` for `np.float32(0.1)`). Lossless, but about 19 characters per value,
  so a 64x64 trajectory with 1000 frames is about 77 MB.
- `init_0.dat` (line 145), `last_0.dat` (line 169, state after `steps` steps, header `step = steps`) and
  `most_recent_0.dat` (line 151, overwritten every `print_last_every`, default `print_trajectory_every`): same
  single-frame format. `last_0.dat` is directly reusable as a `load_from` file.
- `verbose` (default True, line 164) prints every energy line; set `verbose = false` for long runs.

### JAX path (not used here, documented for completeness)

`_write_output_jax_arrays` (lines 250-307): `energy.dat` with times/steps *reconstructed* from the first logged
interval (lines 255-266) and trajectory file `trajectory_species_{i}.dat` with header
`# step = S, species = i, size = NxN` (line 278). The parser in `initial_conditions.read_frames` accepts both headers.

### Known pitfalls found while reading

- `print_trajectory_strategy = "log"` is broken on the CPU path: `_print_trajectory_every` is only set for the linear
  strategy (line 17) but `_run_cpu` reads it unconditionally (line 137), giving `AttributeError`; `print_last_every`
  also defaults to `None` for "log" (line 24) and `None > 0` fails at line 102. Linear output is used.
- `seed` (lines 33-35) only feeds the default `initial_density` noise; with `load_from` it has no effect on the run
  (the Euler update is deterministic).
- `ExplicitEuler` prints a "stability term" `2*M*k*dt/dx**4` (`explicit_euler.py` lines 13-21); this is a rough
  heuristic, not the actual stability bound (see REPORT.md).
- `N` must be a power of 2 (`base_integrator.py` line 46).

## 4. Running and import order

`pfm.main.run(config_filepath, override_use_jax=False, custom_init_fn=None, custom_energy_fn=None)`
(`pfm/main.py` lines 6-32) loads the TOML, sets `CUDA_VISIBLE_DEVICES` / `XLA_PYTHON_CLIENT_MEM_FRACTION` env vars
before importing jax (lines 12-16), enables `jax_enable_x64` if `float_type == "float64"` (lines 19-21), and only then
imports `SimulationManager` (line 24). Order matters because x64 must be enabled before any 64-bit `jnp` array is
created (otherwise JAX silently downcasts to 32 bit), and the env vars are only read when the jax backend
initialises. The relevant arrays are created in constructors, e.g. `landau.py` line 12 (`epsilon`),
`free_energy_model.py` lines 8-9, `explicit_euler.py` lines 10-11 (`dx`, `dt`); a grep of `pfm/utils` and the
`__init__` files finds no module-level `jnp` arrays. Caveat: `pfm/__init__.py` imports `.manager` eagerly, so
`import pfm` already imports `jax.numpy` (but creates no arrays). This study uses `float32`, so x64 ordering does not
affect it; each simulation is still launched in a fresh subprocess calling `pfm.run(...)`.

Update rule actually integrated (`explicit_euler.py` lines 30-45; `base_integrator.py` lines 6-19 for the Laplacian):

```
lap(phi) = sum over axes of [roll(phi,+1) + roll(phi,-1)] - 2*dim*phi, divided by dx**2   (periodic)
mu       = (-eps*rho + rho**3) - interface_scalar * k * lap(rho)
rho_new  = rho + M * dt * lap(mu)
```

With `interface_scalar = 2`, `mu` is exactly the variational derivative of the logged energy
`f(rho) + k|grad rho|^2`, so the dynamics are consistent with the logged free energy.

## 5. Round-trip test output (`python roundtrip_test.py`, N = 32)

```
[ok] default IC: shape (1, 32, 32), dtype float32, formula reproduced; max at (i,j)=(21, 24) (corner-centred rings)
[ok] default IC round-trip through load_from is bit-exact
[ok] asymmetric round-trip bit-exact; Gaussian peak at (i,j)=(8, 21), intended (8, 21); file line i == rho[0, i, :]
[ok] custom_initial_condition path gives identical array
[info] files written by CPU path: ['energy.dat', 'init_0.dat', 'last_0.dat', 'most_recent_0.dat', 'trajectory_0.dat']
[info] trajectory steps [0, 1], times [0.0, 0.001], frames (2, 32, 32)
[info] energy.dat rows: t=[0.0, 0.001] F=[-0.01503284, -0.01503342] mass=[-0.1845, -0.1845] step=[0, 1]
[ok] trajectory frame 0 and init_0.dat equal the IC exactly; last_0.dat shape (1, 32, 32)
[ok] frame 1 matches hand-computed Euler step, max |diff| = 7.62e-09 (float32 rounding)
ALL ROUND-TRIP CHECKS PASSED
```

With `steps = 2` the trajectory holds steps 0 and 1 only and `energy.dat` holds steps 0 and 1; step 2 is only in
`last_0.dat`. The analysis therefore appends `last_0.dat` as the final frame.
