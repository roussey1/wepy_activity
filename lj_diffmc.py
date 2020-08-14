import sys
from copy import copy
import os
import os.path as osp
import pickle
import random

import numpy as np

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import simtk.openmm as omm
import simtk.unit as unit

from testsystems import LennardJonesPair
import mdtraj as mdj
from wepy.util.mdtraj import mdtraj_to_json_topology

from wepy.sim_manager import Manager

from distances.one_prop_dist import OnePropertyDistance
from activities.openmm import LJHarmonicPotential
from diff_mc.diff_util import get_branch_nums
from diff_mc.diff_util import get_entropy
from diff_mc.diff_mc_resampler import DiffMCResampler
from diff_mc.diff_mc_resampler import DiffMCImportance
from forces.harmonic import HarmonicForce

from wepy.walker import Walker
from wepy.runners.openmm import OpenMMRunner, OpenMMState, OpenMMGPUWorker
from runners.new_runner import OpenMMHarmonicPotentialRunner
from wepy.runners.openmm import UNIT_NAMES, GET_STATE_KWARG_DEFAULTS
from wepy.work_mapper.mapper import Mapper
from wepy.work_mapper.mapper import WorkerMapper

from wepy.reporter.hdf5 import WepyHDF5Reporter
import logging

# set logging threshold
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

# Platform used for OpenMM which uses different hardware computation
# kernels. Options are: Reference, CPU, OpenCL, CUDA.
PLATFORM = 'Reference'

# Monte Carlo Barostat
PRESSURE = 1.0*unit.atmosphere
TEMPERATURE = 300.0*unit.kelvin
FRICTION_COEFFICIENT = 1/unit.picosecond
STEP_SIZE = 0.002*unit.picoseconds

# the maximum weight allowed for a walker
PMAX = 0.5
# the minimum weight allowed for a walker
PMIN = 1e-100

# reporting parameters

# these are the properties of the states (i.e. from OpenMM) which will
# be saved into the HDF5
SAVE_FIELDS = ('positions', 'box_vectors', 'velocities', 'activity')
# these are the names of the units which will be stored with each
# field in the HDF5
UNITS = UNIT_NAMES
ALL_ATOMS_SAVE_FREQ = 5

## INPUTS/OUTPUTS

#Read the inputs
if sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("arguments: n_cycles, n_steps, n_walkers, hdf5_filename, epsilon")
    exit(0)
else:
    n_cycles = int(sys.argv[1])
    n_steps = int(sys.argv[2])
    n_walkers = int(sys.argv[3])
    hdf5_filename = sys.argv[4]
    epsilon = int(sys.argv[5])
    
# the inputs directory
inputs_dir = osp.realpath('./inputs')
# the outputs path
outputs_dir = osp.realpath(f'./outputs/diffmc/{epsilon}_eps/entropy0/')

# inputs filenames
json_top_filename = "pair.top.json"

# normalize the input paths
json_top_path = osp.join(inputs_dir, json_top_filename)

omm_states = []
get_state_kwargs = dict(GET_STATE_KWARG_DEFAULTS)
random_pos = random.sample(range(1000), n_walkers)
walker_pos = pickle.load(open(f'inputs/{epsilon}eps_init_positions.pkl', 'rb'))

for i in range(n_walkers):
    # initiate the test system
    test_sys = LennardJonesPair(epsilon=epsilon*unit.kilocalories_per_mole)
    
    # change the box size
    # Vectors set in Vec3. unit=nanometers
    test_sys.system.setDefaultPeriodicBoxVectors([4,0,0],[0,4,0],[0,0,4])
    system = test_sys.system
    omm_topology = test_sys.topology
    mdj_top = mdj.Topology.from_openmm(omm_topology)
    json_top = mdtraj_to_json_topology(mdj_top)
    
    # make the integrator and barostat
    integrator = omm.LangevinIntegrator(TEMPERATURE, FRICTION_COEFFICIENT, STEP_SIZE)

    # build the harmonic force
    total_lj = CustomBondForce("0.5*k*(r-r0)^2")
    total_lj.addPerBondParameter('k')
    total_lj.addPerBondParameter('r0')
    # param [0] is the spring constant, param [1] is the init distance (nm)
    total_lj.addBond(0, 1, [2000, 0.320*unit.nanometer])
    system.addForce(total_lj)
    
    # load positions (randomy selects unrepreated position pairs from
    # a previously generated, well equilibrated system)
    # make a context and set the positions
    context = omm.Context(test_sys.system, integrator)
    test_sys.positions = walker_pos[random_pos[i]]*unit.nanometers
    context.setPositions(test_sys.positions)

    omm_states.append(context.getState(**get_state_kwargs))

    # initialize the runner; get necessary force values
    delta_dist = 0.00336 #nm, distace to go from 0.32nm to 2nm over 500 cycles 

    per_bond_params = total_lj.getBondParameters(0)
    k_init = per_bond_params[2][0]
    print("k_init is ",k_init)
    r0_init = per_bond_params[2][1]
    print("r0_init is ",r0_init)

omm_topology = test_sys.topology
mdj_top = mdj.Topology.from_openmm(omm_topology)
runner = OpenMMHarmonicPotentialRunner(system, omm_topology,
                                       integrator, platform=PLATFORM, enforce_box=True,
                                       activity_metric=LJHarmonicPotential(topology=mdj_top, periodic_state=True),
                                       harmonic_force=HarmonicForce(delta_dist=delta_dist,
                                                                    k_init=k_init, r0_init=r0_init))

distance = OnePropertyDistance('activity')

# get the data from this context so we have a state to start the
# simulation with
init_state = OpenMMState(omm_states[0], activity=np.array([0.]))

## Resampler
Kb = 0.00831446 #kj/(mol*K)
#Amp_Fac = float(Amp_fac)
Beta = 1/(TEMPERATURE*Kb)
ent_cutoff = 0
Imp = DiffMCImportance(beta=Beta)
resampler = DiffMCResampler(imp=Imp, ent_cutoff=ent_cutoff)

## Reporters
# make a dictionary of units for adding to the HDF5
units = dict(UNIT_NAMES)

mapper = Mapper()

## Run the simulation

if __name__ == "__main__":

        
        # normalize the output paths
        hdf5_path = osp.join(outputs_dir, hdf5_filename)


        print("Number of steps: {}".format(n_steps))
        print("Number of cycles: {}".format(n_cycles))
        # # create the initial walkers
        init_weight = 1.0 / n_walkers
        init_walkers = [Walker(OpenMMState(omm_states[i], activity=np.array([0.])), init_weight) for i in range(n_walkers)]


        hdf5_reporter = WepyHDF5Reporter(file_path=hdf5_path, mode='w',
                                         # save_fields set to None saves everything
                                         save_fields=None,
                                         resampler=resampler,
                                         boundary_conditions=None,
                                         topology=json_top,
                                         units=units)
        reporters = [hdf5_reporter]

        sim_manager = Manager(init_walkers,
                              runner=runner,
                              resampler=resampler,
                              boundary_conditions=None,
                              work_mapper=mapper,
                              reporters=reporters)

        # make a number of steps for each cycle. In principle it could be
        # different each cycle
        steps = [n_steps for i in range(n_cycles)]

       # actually run the simulation
        print("Starting run: {}".format(0))
        sim_manager.run_simulation(n_cycles, steps)
        print("Finished run: {}".format(0))


        print("Finished first file")
