import numpy as np
import simtk.openmm as omm
import simtk.unit as unit


class HarmonicForce(object):
    """ A customizable, cycle-index dependent harmonic force. 
        Applicable to a single bond system, AKA a Lennard-Jones Pair."""

    def __init__(self, delta_dist=None, k_init=None, r0_init=None):

        self.delta_dist = delta_dist
        self.k_init = k_init
        self.r0_init = r0_init
        
    def change_force_params(self, cycle_idx, system):

        total_lj = system.getForce(1)
        addition = cycle_idx*self.delta_dist
        params = {'k' : self.k_init, 'd0' : self.r0_init + addition}
        total_lj.setBondParameters(0, 0, 1, [params['k'], params['d0']])

        return system, params

    def get_force_params(self, cycle_idx, system):
        """
        Returns what the value of the parameters would be for 
        a given cycle.
        """

        total_lj = system.getForce(1)
        addition = cycle_idx*self.delta_dist
        params = {'k' : self.k_init, 'd0' : self.r0_init + addition}

        return params
    
