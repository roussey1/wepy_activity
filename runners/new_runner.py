from copy import copy
import random as rand
from warnings import warn
import traceback
import time
import logging

import numpy as np

import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit

from wepy.walker import Walker, WalkerState
from wepy.runners.runner import Runner
from wepy.runners.openmm import OpenMMRunner
from wepy.runners.openmm import OpenMMState
from wepy.work_mapper.worker import Worker
from wepy.reporter.reporter import Reporter


## Constants

# NEW_KEYS = ('positions', 'velocities', 'forces', 'kinetic_energy',
#         'potential_energy', 'time', 'box_vectors', 'box_volume',
#             'parameters', 'parameter_derivatives', 'activity')

# when we use the get_state function from the simulation context we
# can pass options for what kind of data to get, this is the default
# to get all the data. TODO not really sure what the 'groups' keyword
# is for though
GET_STATE_KWARG_DEFAULTS = (('getPositions', True),
                            ('getVelocities', True),
                            ('getForces', True),
                            ('getEnergy', True),
                            ('getParameters', True),
                            ('getParameterDerivatives', True),
                            ('enforcePeriodicBox', True),)

# the Units objects that OpenMM uses internally and are returned from
# simulation data
UNITS = (('positions_unit', unit.nanometer),
         ('time_unit', unit.picosecond),
         ('box_vectors_unit', unit.nanometer),
         ('velocities_unit', unit.nanometer/unit.picosecond),
         ('forces_unit', unit.kilojoule / (unit.nanometer * unit.mole)),
         ('box_volume_unit', unit.nanometer),
         ('kinetic_energy_unit', unit.kilojoule / unit.mole),
         ('potential_energy_unit', unit.kilojoule / unit.mole),
        )

# the names of the units from the units objects above. This is used
# for saving them to files
UNIT_NAMES = (('positions_unit', unit.nanometer.get_name()),
         ('time_unit', unit.picosecond.get_name()),
         ('box_vectors_unit', unit.nanometer.get_name()),
         ('velocities_unit', (unit.nanometer/unit.picosecond).get_name()),
         ('forces_unit', (unit.kilojoule / (unit.nanometer * unit.mole)).get_name()),
         ('box_volume_unit', unit.nanometer.get_name()),
         ('kinetic_energy_unit', (unit.kilojoule / unit.mole).get_name()),
         ('potential_energy_unit', (unit.kilojoule / unit.mole).get_name()),
        )

# a random seed will be chosen from 1 to RAND_SEED_RANGE_MAX when the
# Langevin integrator is created. 0 is the default and special value
# which will then choose a random value when the integrator is created
RAND_SEED_RANGE_MAX = 1000000

class OpenMMHarmonicPotentialRunner(OpenMMRunner):
    #
    # Same as OpenMMRunner, but calculates an "activity" increment (work)
    # using the final frames of each relevant segment, and adds
    # this to the action associated with each state. This differs from the
    # Work runner above by calculating the work based on the potential
    # (U-current and U-future) only at cycles where the lambda parameter,
    # aka the d0 value is modified.
    #

    def __init__(self, system, topology, integrator, platform=None, enforce_box=True, activity_metric=None, harmonic_force=None):

        super().__init__(system, topology, integrator, platform=platform, enforce_box=enforce_box)

        # metric function object
        self.KEYS = ('positions', 'velocities', 'forces', 'kinetic_energy',
                     'potential_energy', 'time', 'box_vectors', 'box_volume',
                     'parameters', 'parameter_derivatives', 'activity')

        # define the activity metric for the system
        self.activity_metric = activity_metric

        # define the time dependent force that will applied to the system
        self.harmonic_force = harmonic_force

        # variables that get checked or changed each cycle
        self._cycle_idx = None
        self._params = {}
                
    def _update_forces(self, cycle_idx):

        # update the forces for running with the updated time dep. forces in the next cycle
        self.system, self._params = self.harmonic_force.change_force_params(cycle_idx, self.system)
        
    def pre_cycle(self, cycle_idx=None, **kwargs):

        super().pre_cycle(**kwargs)

        # get the cycle_idx and current k & d0 values dynamics will be run with  
        self._cycle_idx = cycle_idx
        self._update_forces(cycle_idx)
                
    def generate_state(self, simulation, segment_lengths, starting_walker, getState_kwargs):

        # get the openmm.SimState
        new_sim_state = simulation.context.getState(**getState_kwargs)

        # after running the segment we use the new parameter values
        # to be used in the following cycle in order to calculate
        # the activity increment
        new_params = self.harmonic_force.get_force_params(self._cycle_idx+1, self.system)

        # make an OpenMMState wrapper with new state and updated activity
        old_activity = starting_walker.state['activity']
        
        if self._params != new_params:

            # get the activity increment from the activity function
            activity_incr = self.activity_metric.get_incr(self._params['k'], self._params['d0'],
                                                          OpenMMState(new_sim_state),
                                                          new_params['k'], new_params['d0'])

            # update activity based on a change in d0 (the lambda parameter)
            new_activity = old_activity + activity_incr
        else:
            new_activity = old_activity
            
        # make an OpenMMState wrapper with this
        new_state = OpenMMState(new_sim_state, activity=np.array(new_activity))

        return new_state
