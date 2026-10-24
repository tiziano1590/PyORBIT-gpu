(gpu_usage)=

# Using the GPU

`PyORBIT` can run its per-sample compute (Keplerian solver, transit light curve, Gaussian processes, trends) on an NVIDIA GPU via a JAX backend. This page explains how the GPU mode works, what it can and cannot run, the parameter knobs, a complete worked example with the data it uses, and how to troubleshoot problems.

:::{admonition} CPU-only users
:class: note
If you never set `gpu: true`, none of this applies to you: the GPU gate is a no-op and the code path is identical to a CPU-only install. GPU mode is fully opt-in.
:::

## How the GPU gate works

The whole feature is driven by one YAML flag:

```yaml
parameters:
  gpu: true          # default: false
```

When `gpu: true`, the module `pyorbit.common.gpu_backend` (1) retargets JAX to the CUDA device you select, and (2) runs a **gate** over every model and common object in the fit at setup time:

- models whose class is in `GPU_ALLOWED_MODELS` (`pyorbit/model_definitions.py`) pass;
- everything else **fails fast** with a `GPUSupportedModelError` that lists each offending object and, where a JAX equivalent exists, tells you which YAML keyword to use instead.

There is deliberately **no silent CPU fallback**: if part of your model chain cannot run on the GPU, the run stops at setup with an actionable message rather than quietly running slower on the CPU.

:::{admonition} Why fail fast?
:class: tip
A "GPU" run that secretly evaluates a numba model on the CPU is the worst of both worlds (you pay the JAX overhead and lose the GPU). The gate makes an incompatible fit impossible to start, so you always know exactly where your computation is running.
:::

## Installing the GPU stack

The base package keeps wide JAX bounds so a CPU-only install always resolves. The `[gpu]` extra pins a CUDA-capable stack that actually drives a GPU:

```{code} bash
# CPU-only (default)
pip install -e .

# GPU-enabled
pip install -e .[gpu]
```

`.[gpu]` pins: `jax[cuda12]==0.6.2`, `ml-dtypes<0.6`, `cupy-cuda12x>=13,<14`, `jaxoplanet==0.1.0`, `spleaf==2.1.20`, `batman-package==2.5.3` (batman is the CPU reference kernel).

:::{admonition} Blackwell (RTX PRO 4000/5000/6000)
:class: warning
The *CPU-only* `jaxlib` build (e.g. `jaxlib 0.4.31`) that a plain `pip install -e .` can resolve **cannot** drive a Blackwell GPU (compute capability 12.x). If you see `gpu: true was requested ... no GPU is reachable` or a CUDA driver/library error, you are almost certainly running the CPU-only environment. Use the environment installed with `.[gpu]` (or your equivalent `conda` env with the CUDA JAX stack) for any `gpu: true` run.
:::

Verify the environment before running a fit:

```{code} bash
nvidia-smi                                   # driver + visible VRAM
python -c "import jax; print(jax.devices())"  # must list a CUDA device
```

## Which models run on the GPU

The authoritative list is `GPU_ALLOWED_MODELS` in `pyorbit/model_definitions.py`. In YAML keywords:

| YAML keyword(s) | GPU backend |
|---|---|
| `radial_velocities`, `planetary_velocities`, `rv_planets`, `transit_times` (circular/keplerian/apodized) | JAX-native Keplerian solver (`pyorbit.subroutines.kepler_exo_jax`) |
| `batman_transit` | JAX circular transit (`pyorbit.subroutines.transit_gpu`, jaxoplanet light curve). **Requires a circular orbit and quadratic limb darkening**; anything else fails the gate |
| `spleaf_esp`, `spleaf_multidimensional_esp` (and `_exponentialsineperiodic` aliases) | splEaF (CUDA) |
| `tinygp_quasiperiodic`, `tinygp_quasiperiodic_derivative`, `tinygp_quasiperiodic_cosine`, `tinygp_quasiperiodic_squaredexponential`, `tinygp_sho`, `tinygp_matern32`, `tinygp_multidimensional_*`, `tinygp_multiquasiperiodic_trainedsquaredexponential` | tinygp (JAX-native GPs) |
| `polynomial_trend` | JAX-native Horner evaluation |
| `offset`, `jitter`, dilution / normalization / subset variants | JAX-safe scalar arithmetic |

All common objects (planets, star, limb darkening, activity, ...) are parameter holders only and are always allowed.

Models that are **not** GPU-native but have a JAX equivalent are reported with the keyword to use instead — for example:

| You wrote | Gate says |
|---|---|
| `quasiperiodic` (george) | use `tinygp_quasiperiodic` instead |
| `gp_matern32` (celerite2) | use `tinygp_matern32` instead |
| `spleaf_esp_slow` | use a `tinygp_*` kernel instead |
| `batman_transit` with eccentric orbit / non-quadratic LD | circular + quadratic LD required on the GPU |

Models with no JAX backend in this round (RML fits, TRADES dynamical orbits, starry phase curves, `shared/local/subset_polynomial_trend`, sinusoids, correlation models, CHEOPS detrending, ...) are reported as CPU-only; remove them from a `gpu: true` fit or set `gpu: false`.

## GPU parameters

All live in the YAML `parameters:` section:

```yaml
parameters:
  Tref: 2460000.00
  cpu_threads: 16

  # ---- GPU selection ----
  gpu: true                # enable the JAX/CUDA backend (default: false)
  gpu_device: 0            # which CUDA device to target
  gpu_verbose: true        # print JAX/retargeting diagnostics at setup

  # ---- GPU worker pool (dynesty only, see below) ----
  gpu_workers: 8           # K concurrent dynesty live-point workers (default: 1 = single process)
  gpu_vram_fraction: 0.1   # per-process VRAM cap, as a fraction of the card (default: see below)
```

- `gpu` — the master switch.
- `gpu_device` — index of the CUDA device (JAX devices are enumerated at import time).
- `gpu_verbose` — extra setup diagnostics.
- `gpu_workers` — how many **parallel worker processes** dynesty uses to evaluate its independent live points. Default `1` is the old single-process behaviour.
- `gpu_vram_fraction` — caps each process's JAX memory at `fraction × card_VRAM` (`XLA_PYTHON_CLIENT_MEM_FRACTION`). Default when unset: `min(0.30, max(0.05, 0.90 / (gpu_workers + 1)))`, so with `gpu_workers: 8` it defaults to `0.10` per process.

## Filling the GPU: the dynesty worker pool

**Why it exists.** A transit+GP likelihood is small (tens to thousands of data points, a few dozen parameters). Evaluated one at a time, each `mc(theta)` call is dominated by host↔device round-trips and Python control flow, so the GPU sits idle between evaluations — an A/B test measured the single-process GPU leg *slower* than the CPU for exactly this reason. The fix is to keep the GPU busy with **K concurrent evaluations** that share one card.

**How it works.** When `gpu: true` and `gpu_workers > 1`, `pyorbit_dynesty` hands dynesty's live points to a pool of `gpu_workers` **spawned** worker processes (not forked — a CUDA context cannot be inherited across `fork()`):

- each worker is a fresh interpreter that rebuilds an equivalent `ModelContainerDynesty` from the pickled `config_in` and its own CUDA context;
- the master process keeps one context for the prior transform (dynesty's `use_pool={'prior_transform': False}`), so the prior always runs on the master's copy;
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=<gpu_vram_fraction>` are exported before the workers start, so every process (master included) is VRAM-capped;
- checkpoint/restore is unchanged (dynesty strips the pool on pickling and re-attaches it on `restore`).

Precedence: **GPU spawn pool** (`gpu: true`, `gpu_workers > 1`) → CPU fork pool (`use_threading_pool`, CPU-only runs) → single process.

### VRAM budget

With `gpu_workers: 8` and the default `0.10` per-process fraction, `9` processes (master + 8 workers) × 0.10 = 0.90 of the card is reserved — on a 32 GiB card that is ≈ **28.7 GiB that must be free** before the run starts. Close every other CUDA consumer first:

```{code} bash
nvidia-smi   # look for other processes (e.g. an LLM server like ollama) holding VRAM
```

If the card is smaller, lower the budget accordingly (fewer workers and/or a smaller fraction), e.g. `gpu_workers: 4`, `gpu_vram_fraction: 0.15`.

## Worked example: TOI-1554 (Kepler-63 b)

A complete, self-contained CPU/GPU A/B fit lives in [`development/examples/kepler63b_gpu/`](https://github.com/tiziano1590/PyORBIT-gpu/tree/main/development/examples/kepler63b_gpu). It fits the radial velocities plus two activity indicators (Hα equivalent width, line core depth) with a multi-dataset quasi-periodic activity GP, and two independent light curves (Kepler + TESS) with a transit plus a low-order detrend — the classic "free transit depth" problem.

### The data

| File | Contents | Points | Epochs (BJD) |
|---|---|---|---|
| `RVS_PyORBIT.txt` | epoch, RV (m/s), error, `jitter_flag`, `offset_flag`, `subset_flag` | 62 | ~2460040–2460280 |
| `Halpha_PyORBIT.txt` | epoch, Hα equivalent width, error, flags | 62 | ~2460040–2460280 |
| `EW_PyORBIT.txt` | epoch, line-core depth, error, flags | 62 | ~2460040–2460280 |
| `KEPLER63_KEPLER.DAT` | epoch, relative flux, error, flags (normalized, activity-dominated) | 56,191 | ~2455002–2456424 |
| `KEPLER63_TESS.DAT` | epoch, relative flux, error, flags | 6,484 | ~2458683–2460533 |

### The model chain

- **Activity:** `tinygp_multidimensional_quasiperiodic` on RV + Hα + EW, with `hyperparameters_condition: True` and `rotation_decay_condition: True`, per-dataset `rot_amp`/`con_amp` (and a derivative term on RV/EW). This is the JAX-native GP — on the CPU leg the same config works because the model is JAX-native on CPU too (the CPU/GPU difference is the *backend*, not the math).
- **Transit:** `batman_transit` on each light curve, **circular orbit + quadratic limb darkening** (the two constraints the GPU transit kernel imposes), free `R_Rs` ∈ [0.01, 0.30]. On `gpu: true` this evaluates through the JAX `circular_transit` kernel; on `gpu: false` through numba `batman` — the same physical model.
- **Detrend:** one `polynomial_trend` (order 2) per light curve. A deliberate detrend choice: a QP activity GP over the full 62,858-point light curves would need a ~29 GiB kernel (plus two `jax.grad` passes) — more than the 32 GiB card — and an infeasible O(N³) solve on CPU; a quadratic trend removes the same slow activity drift at O(N) cost and isolates the transit depth.
- **Priors:** all free/internal parameters carry **wide** uniform priors (broadly non-informative), so the data — not tight priors — sets the result.

### The parameter files

Two byte-identical configs, differing **only** in the `gpu` flag:

- `TOI1554_CPU_dynesty_phot_WIDE_Trend.yaml` — `gpu: false` (CPU reference leg)
- `TOI1554_GPU_dynesty_phot_WIDE_Trend.yaml` — `gpu: true`, `gpu_device: 0`, `gpu_workers: 8`, `gpu_vram_fraction: 0.1`

Solver settings (both legs): `nlive: 200`, `bound: multi`, `sample: rwalk`, `dlogz: 0.01`, `maxcall: 40000`, `use_threading_pool: False`, `recenter_bounds: True`.

### Running it

```{code} bash
cd development/examples/kepler63b_gpu

# GPU leg (run in the GPU environment; make sure ~29 GiB of VRAM is free first)
pyorbit_run dynesty TOI1554_GPU_dynesty_phot_WIDE_Trend.yaml
pyorbit_results dynesty TOI1554_GPU_dynesty_phot_WIDE_Trend.yaml -all

# CPU reference leg (same model, numba transit; single process)
pyorbit_run dynesty TOI1554_CPU_dynesty_phot_WIDE_Trend.yaml
pyorbit_results dynesty TOI1554_CPU_dynesty_phot_WIDE_Trend.yaml -all
```

The two legs are a true A/B: same data, same priors, same solver, same model classes — the only differences are the compute backend (numba vs JAX/CUDA) and the dynesty worker pool on the GPU leg.

### Monitoring the GPU while it runs

`check_gpu_usage.sh` (in the same folder) is a read-only live monitor:

```{code} bash
./check_gpu_usage.sh            # 1 s refresh, Ctrl-C to stop
./check_gpu_usage.sh --once     # single snapshot
./check_gpu_usage.sh -n 10      # 10 samples and exit
./check_gpu_usage.sh --log run.log
```

It prints per-process VRAM (so you can see the 8 workers each staying under their cap) and flags any other process holding VRAM (e.g. a local LLM server) that is stealing from the run.

## Troubleshooting

**`GPUSupportedModelError` at setup.** Read the message: it lists each offending model/common object and, where possible, the JAX keyword to use instead (see the table above). Fix the YAML and re-run.

**OOM / "Failed to allocate" / the run dies part-way.** Too many processes × too much per-process VRAM, or another CUDA consumer is present. Lower `gpu_workers` and/or `gpu_vram_fraction`, and close other CUDA applications (`nvidia-smi`).

**`gpu: true` but no GPU is reachable / CUDA driver error.** You are in the CPU-only environment (CPU-only `jaxlib` cannot drive newer GPUs, Blackwell included). Activate the environment installed with `.[gpu]` and re-check `python -c "import jax; print(jax.devices())"`.

**`pyorbit_results ... -all` hangs.** This was a real bug, and it is not GPU-specific: `pyorbit_results` evaluates models in a worker pool, and `os.fork()` after JAX has initialized is a classic deadlock (JAX/XLA runs multithreaded C thread pools that a forked child cannot inherit, and a live CUDA context cannot be inherited at all). It bites `gpu: true` runs *and* CPU-only runs that use JAX-native models such as `tinygp_*`. It is fixed in this branch — when the process is multithreaded, has used the GPU, or has merely loaded JAX, the results workers now start with `spawn` (fresh interpreters) instead of `fork` (you will see a `NOTE: ... using 'spawn' instead of 'fork'` line), and the code still falls back to serial evaluation if the pool cannot be used. If you still see a hang, run `pyorbit_results` in a fresh process (it rebuilds everything from the checkpoint).

**Verifying GPU/CPU parity.** `development/gpu_cpu_parity.py` evaluates the same model at identical parameter values on both backends and reports the max absolute difference; `development/run_kepler63b_gpu_test.sh` drives the full A/B. Expect agreement at float64 round-off for the shared models.
