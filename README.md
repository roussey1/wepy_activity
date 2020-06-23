## About this software

   This package is for use with '[Wepy](https://github.com/ADicksonLab/wepy)', and can be used to calculate (un)binding free energies from pulling trajectories of Lennard-Jones Pairs. This code was used to generate the data in N. M. Roussey & A. Dickson, 2020.    

## Additional info
The theory for this project is based in nonequilibrium stastical mechanics, specifically the Jarzynski Nonequilibrium Work Relation AKA '[The Jarzynski Equality](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.78.2690)'. This is an equation that relates the work done by a nonequilibrium process on a system to the equilibrium free energy difference between the inital and final states. This equation is of interest because it allows us to perform nonequilibrium work (along some system-relevant lambda from lambda_0 to lambda_1) to generate equilibrium free energy differences. The JE is as follows: <exp(-beta W)> = exp(-beta* Delta F), where <...> indicate the ensemble average, W is work, and Delta F is the Helmholtz free energy. This is significant because these nonequilibrium simualtions can be very fast and potentially less computationly expensive than traditional equilibrium simualtions used to determine the same value.
   
   The final free energy determined is dominated by low work / high importance trajectories, and we are interested in determining if enhancing the sampling of low work trajectories may allow us to find an accurate value of Delta F without the necessary number of trajectories otherwise needed for convergence. We have aimed to develop a new method of calculating free energy differences between states of a system utilizing nonequilibrium processes and means of analyzing these nonequilibrium processes. This software conatins a new "importance resampler" as well as a "history-dependent" version of the novelty-REVO resampler. Free energy surfaces can be generated for these simulations with the provided analysis code.
   
## Authors

Nicole Roussey(1), Samuel Lotz(1), & Alex Dickson(1)(2)

(1). The Department of Biochemistry and Molecular Biology, Michigan State University, East Lansing, Michigan

(2). The Department of Computational Mathemetaics, Science, & Engineering, Michigan State University, East Lansing, Michigan

## Installation and Requirements

To use this software:

First clone the necessary git repos.

git clone https://gitlab.com/ADicksonLab/wepy.git

install wepy with "pip install -e ."

git clone https://gitlab.com/nmroussey/lj_activity.git

conda install -c omnia -c conda-forge openmm

conda install -c conda-forge mdtraj

'[Wepy]('https://github.com/ADicksonLab/wepy)' will install most dependencies. 

## Example testing

Then to test the example scripts (containing values for 500 cycle simulations):

python lj_noresampler.py 500 100 50 test_noresampler.wepy.h5 10 #args are num_cycles num_steps num_walkers output_file_name epsilon

python lj_revo.py 500 100 50 test_revo.wepy.h5 10 char_dist* #args are num_cycles num_steps num_walkers output_file_name epsilon *characteristic distance (see: Donyapour, 2019)

*For epsilon = 10 kcal/mol, char_dist is 0.46396780

python lj_importance.py 500 100 50 test_importance.wepy.h5 10 1.0 #args are num_cycles num_steps num_walkers output_file_name epsilon amplification

## Analysis testing

The analysis provided here is equation 8 from Hummer and Szabo, 2001. This creates a free energy surface from the probabilities and work values saved during a simulation.

To run this analysis code:

1. run the REVO example above

2. cd analysis/

3. python hz_equation8.py 10 50 ../outputs/resampler/10_eps/test_revo.wepy.h5 #args are epsilon num_walkers path/to/file

A plot of the target free energy surface for epsilon 10 as well as the free energy surface generated from the simulation data will appear.

## References

Software Packages

'[Wepy](https://github.com/ADicksonLab/wepy)'

'[OpenMM](http://openmm.org)'

'[mdtraj](http://mdtraj.org/1.9.3/)'

'[NumPy](https://numpy.org)'

'[h5py](https://www.h5py.org)'

Papers

'[Nonequilibrium Equality for Free Energy Differences](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.78.2690)' Jarzynski, C., PRL, 1997

Analysis from:

'[Free energy reconstruction from nonequilibrium single-molecule pulling experiments](https://www.pnas.org/content/98/7/3658)' Hummer, G., & Szabo, A., PNAS, 2001