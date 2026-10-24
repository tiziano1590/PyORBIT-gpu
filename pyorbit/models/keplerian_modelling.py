from __future__ import print_function

from more_itertools import prepend
from pyorbit.subroutines.common import *
from pyorbit.models.abstract_model import AbstractModel

import pyorbit.subroutines.kepler_exo as kepler_exo

from scipy.stats import norm

from types import SimpleNamespace

import pyorbit.common.gpu_backend as gpu_backend


def _active_kepler():
    """Dispatch the Keplerian / impact-parameter / apodization calls to the
    active backend (Round 1 GPU enablement).

    Returns a small namespace exposing the functions this module calls:

        rv_deltabjd, rv_semiamplitude, meanlong_from_dTc,
        deltaTc_from_meanlong, convert_rho_to_ars, convert_b_to_i, norm_pdf

    * ``gpu: true``  -> the JAX variants from
      ``pyorbit.subroutines.kepler_exo_jax`` and ``jax.scipy.stats.norm.pdf``.
      ``kepler_exo_jax`` is imported **lazily** here so the CPU path never loads
      a JAX-dependent module.
    * ``gpu: false`` (the default) -> the numpy ``kepler_exo`` functions and
      ``scipy.stats.norm.pdf`` exactly as before, so the CPU path is unchanged.

    The two sets are numerically identical under x64 (see
    ``development/gpu_cpu_parity.py``), so results do not depend on the backend.
    """
    if gpu_backend.gpu_enabled():
        import jax
        import pyorbit.subroutines.kepler_exo_jax as kepler_exo_jax
        return SimpleNamespace(
            rv_deltabjd=kepler_exo_jax.kepler_compute_rv_deltabjd_jax,
            rv_semiamplitude=kepler_exo_jax.kepler_compute_rv_semiamplitude_jax,
            meanlong_from_dTc=kepler_exo_jax.kepler_compute_meanlong_from_deltaTc_jax,
            deltaTc_from_meanlong=kepler_exo_jax.kepler_compute_deltaTc_from_meanlong_jax,
            convert_rho_to_ars=kepler_exo_jax.convert_rho_to_ars_jax,
            convert_b_to_i=kepler_exo_jax.convert_b_to_i_jax,
            norm_pdf=jax.scipy.stats.norm.pdf,
        )
    return SimpleNamespace(
        rv_deltabjd=kepler_exo.kepler_compute_rv_deltabjd,
        rv_semiamplitude=kepler_exo.kepler_compute_rv_semiamplitude,
        meanlong_from_dTc=kepler_exo.kepler_compute_meanlong_from_deltaTc,
        deltaTc_from_meanlong=kepler_exo.kepler_compute_deltaTc_from_meanlong,
        convert_rho_to_ars=convert_rho_to_ars,
        convert_b_to_i=convert_b_to_i,
        norm_pdf=norm.pdf,
    )


class RVkeplerian(AbstractModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_class = 'radial_velocities'

        self.list_pams_common = OrderedSet([
            'P',  # Period
            'K',  # RV semi-amplitude
            'e',  # eccentricity, uniform prior - to be fixed
            'omega'  # argument of pericenter
        ])

        self.use_time_inferior_conjunction = False
        self.use_mass = False
        self.use_scaled_mass = False
        self.use_stellar_scaled_mass = False

        """ Default is to use the inclination value as provided.
        Otherwise, compute it using the semimajor axis derived from orbital period and stellar density"""
        self.compute_inclination = False
        self.compute_scaled_semimajor_axis = True
        self.compute_semimajor_axis = True

    def initialize_model(self, mc, **kwargs):

        # Copying the property to the class for faster access
        self.use_time_inferior_conjunction = getattr(mc.common_models[self.planet_ref], 
                                                    'use_time_inferior_conjunction', self.use_time_inferior_conjunction)
        self.use_mass = getattr(mc.common_models[self.planet_ref], 'use_mass', self.use_mass)
        self.use_scaled_mass = getattr(mc.common_models[self.planet_ref], 'use_scaled_mass', self.use_scaled_mass)
        self.use_stellar_scaled_mass = getattr(mc.common_models[self.planet_ref], 'use_stellar_scaled_mass', self.use_stellar_scaled_mass)
        self.compute_inclination = getattr(mc.common_models[self.planet_ref], 'compute_inclination', self.compute_inclination)
        self.compute_scaled_semimajor_axis = getattr(mc.common_models[self.planet_ref], 'compute_scaled_semimajor_axis', self.compute_scaled_semimajor_axis)

        if mc.common_models[self.planet_ref].parametrization[:8] == 'Ford2006' \
            and mc.common_models[self.planet_ref].orbit != 'circular':
            self.list_pams_common.discard('e')
            self.list_pams_common.discard('omega')

            self.list_pams_common.update(['e_coso'])
            self.list_pams_common.update(['e_sino'])

        elif mc.common_models[self.planet_ref].parametrization[:8] != 'Standard' \
            and mc.common_models[self.planet_ref].orbit != 'circular':
                # 'Eastman2013' is the standard choice
            self.list_pams_common.discard('e')
            self.list_pams_common.discard('omega')

            self.list_pams_common.update(['sre_coso'])
            self.list_pams_common.update(['sre_sino'])



        if self.use_time_inferior_conjunction:
            self.list_pams_common.update(['Tc'])
        else:
            self.list_pams_common.update(['mean_long'])

        if self.use_mass or self.use_scaled_mass or self.use_stellar_scaled_mass:
            self.list_pams_common.discard('K')

            if self.use_mass:
                self.list_pams_common.update(['M_Me'])
            elif self.use_scaled_mass:
                self.list_pams_common.update(['Me_Ms'])
            elif self.use_stellar_scaled_mass:
                self.list_pams_common.update(['M_Ms'])
            self.list_pams_common.update(['mass'])

            if self.compute_inclination:
                """ b is the impact parameter """
                self.list_pams_common.update(['b'])
                if self.compute_scaled_semimajor_axis:
                    self.list_pams_common.update(['density'])
                else:
                    self.list_pams_common.update(['a_Rs'])
            else:
                """ i is the orbital inclination (in degrees) """
                self.list_pams_common.update(['i'])
        else:
            self.list_pams_common.update(['K'])


        if mc.common_models[self.planet_ref].use_longitude_of_nodes:
            self.list_pams_common.update(['Omega'])

        try:
            self.default_Omega = mc.common_models[self.planet_ref].default_Omega
        except AttributeError:
            self.default_Omega = 0.00

    def compute(self, parameter_values, dataset, x0_input=None):

        kx = _active_kepler()

        Omega = parameter_values.get('Omega', self.default_Omega)

        if self.use_time_inferior_conjunction:
            mean_long = kx.meanlong_from_dTc(parameter_values['P'],
                                                parameter_values['Tc'] - dataset.Tref,
                                                parameter_values['e'],
                                                parameter_values['omega'],
                                                Omega)
        else:
            mean_long = parameter_values['mean_long']

        if self.use_mass or self.use_scaled_mass or self.use_stellar_scaled_mass:

            if self.compute_inclination:
                if self.compute_scaled_semimajor_axis:
                    parameter_values['a_Rs'] = kx.convert_rho_to_ars(parameter_values['P'],
                                                                    parameter_values['density'])

                parameter_values['i'] = kx.convert_b_to_i(
                    parameter_values['b'],
                    parameter_values['e'],
                    parameter_values['omega'],
                    parameter_values['a_Rs'])

            if self.use_mass:
                planet_mass = parameter_values['M_Me'] / constants.Msear
            elif self.use_scaled_mass:
                planet_mass = parameter_values['Me_Ms'] * parameter_values['mass'] / constants.Msear
            elif self.use_stellar_scaled_mass:
                planet_mass = parameter_values['M_Ms'] * parameter_values['mass']

            rv_semiamplitude = kx.rv_semiamplitude(parameter_values['mass'],
                                                                            planet_mass,
                                                                            parameter_values['P'],
                                                                            parameter_values['i'],
                                                                            parameter_values['e'])

        else:
            rv_semiamplitude = parameter_values['K']

        if x0_input is None:
            return kx.rv_deltabjd(dataset.x0,
                                            rv_semiamplitude,
                                            parameter_values['P'],
                                            mean_long,
                                            parameter_values['e'],
                                            parameter_values['omega'],
                                            Omega)
        else:
            return kx.rv_deltabjd(x0_input,
                                            rv_semiamplitude,
                                            parameter_values['P'],
                                            mean_long,
                                            parameter_values['e'],
                                            parameter_values['omega'],
                                            Omega)



class ApodizedRVkeplerian(RVkeplerian):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.list_pams_common.update(['apo_center'])
        self.list_pams_common.update(['apo_timescale'])

    def initialize_model_dataset(self, mc, dataset, **kwargs):

        try:
            min_apo_center = min(mc.common_models[self.planet_ref].default_bounds['apo_center'][0],
                                    np.min(dataset.x))
            max_apo_center = max(mc.common_models[self.planet_ref].default_bounds['apo_center'][1],
                                    np.max(dataset.x))
        except KeyError:
            min_apo_center = np.min(dataset.x)
            max_apo_center = np.max(dataset.x)

        mc.common_models[self.planet_ref].default_bounds['apo_center'] = [min_apo_center, max_apo_center]
        return

    def compute(self, parameter_values, dataset, x0_input=None):

        kx = _active_kepler()

        Omega = parameter_values.get('Omega', self.default_Omega)

        if self.use_time_inferior_conjunction:
            mean_long = kx.meanlong_from_dTc(parameter_values['P'],
                                                parameter_values['Tc'] - dataset.Tref,
                                                parameter_values['e'],
                                                parameter_values['omega'],
                                                Omega)
        else:
            mean_long = parameter_values['mean_long']

        if self.use_mass or self.use_scaled_mass or self.use_stellar_scaled_mass:

            if self.use_mass:
                planet_mass = parameter_values['M_Me'] / constants.Msear
            elif self.use_scaled_mass:
                planet_mass = parameter_values['Me_Ms'] * parameter_values['mass'] / constants.Msear
            elif self.use_stellar_scaled_mass:
                planet_mass = parameter_values['M_Ms'] * parameter_values['mass']

            rv_semiamplitude = kx.rv_semiamplitude(parameter_values['mass'],
                                                                            planet_mass,
                                                                            parameter_values['P'],
                                                                            90.0,
                                                                            parameter_values['e'])
        else:
            rv_semiamplitude = parameter_values['K']

        if x0_input is None:

            # Apodization factor
            apo_function = (kx.norm_pdf(dataset.x,
                                        parameter_values['apo_center'],
                                        parameter_values['apo_timescale'])
                            / kx.norm_pdf(parameter_values['apo_center'],
                                        parameter_values['apo_center'],
                                        parameter_values['apo_timescale']))

            return apo_function * kx.rv_deltabjd(dataset.x0,
                                            rv_semiamplitude,
                                            parameter_values['P'],
                                            mean_long,
                                            parameter_values['e'],
                                            parameter_values['omega'],
                                            Omega)
        else:

            # Apodization factor
            apo_function = (kx.norm_pdf(dataset.x,
                                        parameter_values['apo_center'],
                                        parameter_values['apo_timescale'])
                            / kx.norm_pdf(parameter_values['apo_center'],
                                        parameter_values['apo_center'],
                                        parameter_values['apo_timescale']))

            return apo_function * kx.rv_deltabjd(dataset.x0,
                                            rv_semiamplitude,
                                            parameter_values['P'],
                                            mean_long,
                                            parameter_values['e'],
                                            parameter_values['omega'],
                                            Omega)


class TransitTimeKeplerian(AbstractModel):
    model_class = 'transit_times'

    def __init__(self, *args, **kwargs):
        super(TransitTimeKeplerian, self).__init__(*args, **kwargs)
        self.use_time_inferior_conjunction = False

        self.list_pams_common = OrderedSet(['P'])  # Period

        self.list_pams_dataset = OrderedSet()

    def initialize_model(self, mc, **kwargs):

        if mc.common_models[self.planet_ref].use_time_inferior_conjunction:
            self.list_pams_common.update(['Tc'])
            self.use_time_inferior_conjunction = True
            # Copying the property to the class for faster access
        else:
            self.list_pams_common.update(['mean_long'])
            self.list_pams_common.update(['e'])
            self.list_pams_common.update(['omega'])
            # mean longitude = argument of pericenter + mean anomaly at Tref

        try:
            self.default_Omega = mc.common_models[self.planet_ref].default_Omega
        except AttributeError:
            self.default_Omega = 0.00


    def compute(self, parameter_values, dataset, x0_input=None):

        kx = _active_kepler()

        #TODO check this function, I believe it's a bit crazy.... URGENT
        Omega = parameter_values.get('Omega', self.default_Omega)

        if self.use_time_inferior_conjunction:
            delta_T = parameter_values['Tc'] - \
                np.floor((parameter_values['Tc'] - dataset.Tref) / parameter_values['P']) * parameter_values['P']
        else:
            delta_T = dataset.Tref + \
                      kx.deltaTc_from_meanlong(parameter_values['P'],
                                                      parameter_values['mean_long'],
                                                      parameter_values['e'],
                                                      parameter_values['omega'],
                                                      Omega)

        if x0_input is None:
            return np.floor(dataset.x0 / parameter_values['P']) * parameter_values['P'] + delta_T
        else:
            return np.floor(x0_input / parameter_values['P']) * parameter_values['P'] + delta_T


