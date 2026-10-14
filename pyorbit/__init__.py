#!/usr/bin/env python
# -*- coding: utf-8 -*-

# GPU enablement (Round 1) -- pre-import hook.
#
# If the launcher (``python PyORBIT_Run.py``, or a user's own script) has already
# decided this is a GPU run, it can set ``PYORBIT_GPU=1`` *before* importing
# pyorbit. We honor that here, before any of the modules below -- which
# transitively import jaxlib and lock in its default platform -- execute, so that
# jaxlib itself comes up on the GPU.
#
# This is purely optional insurance: the authoritative enablement, the
# GPU-reachability check and the actionable error all live in
# ``pyorbit.common.gpu_backend.initialize()``, which runs after the config is
# parsed. When ``PYORBIT_GPU`` is unset (the default, i.e. ``gpu: false``) this
# block does nothing and the CPU path is byte-for-byte unchanged.
import os as _pyorbit_os
if _pyorbit_os.environ.get('PYORBIT_GPU', '') not in ('', '0', 'false', 'False', 'no', 'off'):
    _pyorbit_os.environ.setdefault('JAX_PLATFORMS', 'gpu')
    _pyorbit_os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')
    _pyorbit_os.environ.setdefault('JAX_ENABLE_X64', 'true')

from .pyorbit_run import *
from .pyorbit_results import *
from .model_definitions import *
from .datatype_definitions import *
from .samplers.pyorbit_emcee import *
from .samplers.pyorbit_emcee_legacy import *
from .samplers.pyorbit_emcee_mpi import *
from .samplers.pyorbit_zeus_legacy import *
from .samplers.pyorbit_optimize import *
from .samplers.pyorbit_polychord import *
from .samplers.pyorbit_pyde import *
from .samplers.pyorbit_dynesty import *
from .samplers.pyorbit_dynesty_static import *
from .samplers.pyorbit_dynesty_legacy import *
from .samplers.pyorbit_nestle import *
from .samplers.pyorbit_nautilus import *
from .samplers.pyorbit_ultranest import *
from .samplers.pyorbit_ultranest_stepsampler import *
from .samplers.pyorbit_ultranest_warmstart import *
from .samplers.pyorbit_multinest import *
from .samplers.pyorbit_getresults import *
from .subroutines.input_parser import yaml_parser


__version__ = "11.2.8"
