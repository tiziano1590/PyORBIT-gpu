# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyORBIT is a Bayesian fitting framework for exoplanet orbital parameters and stellar activity. It simultaneously fits multiple datasets (radial velocities, transit light curves, activity indicators, CCFs, astrometry) with a mix of models (Keplerian/dynamical orbits, transits, Gaussian processes, sinusoids, trends, Rossiter-McLaughlin) using MCMC or nested sampling. This repository is the `PyORBIT-gpu` fork of `LucaMalavolta/PyORBIT` (v11.2.8).

## Commands

```bash
# Install (development)
pip install -e .
pip install -r extra_requirements.txt   # optional extras: tinygp, celerite2, dynesty, ultranest, starry, PyDE, ...

# Run an analysis (entry point declared in pyproject.toml [project.scripts])
pyorbit_run <sampler> <config_file.yaml>
# equivalent: python PyORBIT_Run.py <sampler> <config_file.yaml>
# samplers: pyde, optimize, emcee, emcee_mpi, zeus_legacy, dynesty, dynesty_static,
#           dynesty_dryrun, nestle, ultranest, ultranest_stepsampler, ultranest_warmstart,
#           nautilus, multinest, polychord

# Analyze results (corner plots, ACF, model files, etc.)
pyorbit_results <sampler> <config_file.yaml> [flags]
# common flags: -all (everything), -p (plot models), -w (write model files), -c (chains),
#               -a (ACF), -t (Gelman-Rubin), -fc (full correlation), -cc/-dc (common/dataset
#               corner plots), -ws/-wp (write all/planet samples), -compute_rv_like
```

- There is **no test suite and no CI**; `development/` holds ad-hoc development scripts and Jupyter notebooks (not tests).
- Documentation is Sphinx: `cd docs && make html` (Read the Docs builds with the `dirhtml` builder, config in `.readthedocs.yaml`).
- `fetch_constants.sh` downloads TRADES constants into `pyorbit/classes/constants.pyx` (note the odd `.pyx` extension; it is plain Python imported as `pyorbit.subroutines.constants`).

## Architecture

### Pipeline

1. A YAML config file has five sections: `inputs` (datasets), `common` (shared objects: planets, star, limb darkening, activity), `models` (per-model configuration), `parameters` (e.g. `Tref`, `cpu_threads`), `solver` (per-sampler settings).
2. `pyorbit/pyorbit_run.py` dispatches on the sampler keyword (alias table `sampler_keyword`) to `pyorbit.<sampler>(config_in)` in `pyorbit/samplers/`.
3. Each sampler module: `yaml_parser()` → `pars_input(config_in, mc)` (`pyorbit/subroutines/input_parser.py`) instantiates a `ModelContainer*` subclass and populates it from the YAML via the registries → `mc.model_setup()` → `mc.boundaries_setup()` → `mc.initialize_logchi2()` → the sampler minimizes/samples `mc(theta)` (log priors + log likelihood) → results pickled to `./<output>/<sampler>/`.
4. Re-running the same command **resumes**: samplers attempt to load previous checkpoints (`*load_from_cpickle` in `pyorbit/subroutines/io_subroutines.py`) and continue or print results. The typical emcee flow is PyDE (differential evolution) first, then emcee.
5. `pyorbit_results` (→ `pyorbit/samplers/pyorbit_getresults.py`) reloads checkpoints and writes statistics/plots/model files.

### Core object model

- **`ModelContainer`** (`pyorbit/classes/model_container_abstract.py`) — the central object. Owns `dataset_dict`, `common_models`, `models`, `bounds`, `spaces`, `priors`. `__call__(theta)` and `log_priors_likelihood(theta)` compute the full log probability; `rv_log_likelihood(theta)` sums per-dataset Gaussian log-likelihoods; `log_priors(theta)` sums per-parameter priors.
- **Per-sampler subclasses** (`pyorbit/classes/model_container_*.py`) only add sampler parameter dicts (e.g. `emcee_parameters`, `pyde_parameters`) with their defaults; the YAML `solver:` section overrides them.
- **`AbstractModel`** (`pyorbit/models/abstract_model.py`) — base for all observation models. The key distinction is `list_pams_common` (parameters shared across all datasets using the model, e.g. GP period — stored in a common object) vs `list_pams_dataset` (per-dataset parameters, e.g. GP amplitude). `common_ref` links a model to the common objects it depends on (a planet, the star, limb darkening). Each model implements `compute(parameter_values, dataset)` and `initialize_model*` methods. `unitary_model` / `normalization_model` flag special multiplicative models.
- **`AbstractCommon`** (`pyorbit/common/abstract_common.py`) — base for shared objects instantiated under the YAML `common:` section: `CommonPlanets`, `CommonStarParameters`, limb darkening laws, `CommonActivity`, `CommonCorrelation`, etc.
- **`Dataset`** (`pyorbit/common/dataset.py`) — wraps one input file (columns: epoch, value, error, then `jitter_flag`/`offset_flag`/`subset_flag`; `transit_duration` files have one extra data column) plus the list of models applied to it. `compute(variable_value)` combines the model outputs; `model_logchi2()` gives the dataset's contribution to the likelihood. `DatasetExpanded` handles multidimensional (2D) datasets.

### Registries (critical for adding or renaming things)

- **`pyorbit/model_definitions.py`** — `define_type_to_class` maps YAML model keywords (many aliases) to model classes; `define_common_type_to_class` maps `common:` keywords to common classes; plus requirement lists (`model_requires_planets`, `model_requires_limb_darkening`, `model_requires_star`, `single_planet_model`, ...). **To add a new model: create the class in `pyorbit/models/`, then import and register it here.**
- **`pyorbit/datatype_definitions.py`** — `datatype_definition` maps dataset `kind` aliases (e.g. `RV`, `Tc`, `phot`) to canonical kinds; `activity_datatype` lists activity-indicator kinds.
- **`pyorbit/keywords_definitions.py`** — alias lists for boolean/keyword YAML options (e.g. `use_Prot` / `use_activity_Prot`).

### Parameter spaces

Parameters are sampled in transformed spaces: `Linear`, `Log_Natural`, `Log_Base2`, `Log_Base10` (forward/inverse via `get_var_log/exp*` in `pyorbit/subroutines/common.py`). Priors: `Uniform` (default), `Gaussian`, `Jeffreys`, `Truncated Rayleigh`, `BetaDistribution`, plus custom multivariate priors. MCMC and nested samplers handle priors differently — a prior valid for one may not be for the other.

### Notable conventions and gotchas

- Version 11 behavior change (v11.2+): mean longitude = ω + Ω + M₀ at `Tref`, with default Ω = 180°. Results from v10 are not comparable.
- `deprecated/` contains superseded model implementations — do not use or build on them.
- Packaging moved to `pyproject.toml` in v12 (upstream deleted `setup.py`). Base `dependencies` keep wide bounds (`numpy>=1.24.3,<2.0`, `jax>=0.4.31,<=0.6.2`, `tinygp==0.3.0`; tinygp 0.3.0 requires Python 3.10). The `[gpu]` extra (`pip install pyorbit-package[gpu]`) pins the CUDA JAX stack (`jax[cuda12]==0.6.2`) plus the JAX-native kernels (jaxoplanet, splEaF, cuPy, batman). `extra_requirements.txt` still lists looser pins for optional samplers — install the base package first.
- The PyPI distribution is named `pyorbit-package` (the bare name `pyorbit` is taken by another package); the import name is `pyorbit`.
- `PyORBIT_GetResults.py` is a symlink to `PyORBIT_Results.py`; both are thin wrappers around `pyorbit.pyorbit_results()`.
- `pyorbit/__init__.py` is where sampler modules are exposed as `pyorbit.pyorbit_<sampler>`; new samplers must be added there and to the `sampler_keyword` table in `pyorbit_run.py` (and the matching table in `pyorbit_results.py`).
