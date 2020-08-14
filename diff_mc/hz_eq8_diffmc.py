# analysis from the hummer szabo paper using equation 8, the neqWHAM

import random
from wepy.hdf5 import WepyHDF5
import mdtraj as mdj
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import pi
#from numba import jit 

d0_start = 0.32 #nm
d0_incr = 0.00336 #nm
n_cycles = 500
d_min = 0
d_max = 8

#### jarzynski analysis
T = 300 #K
Kb = 0.008314 #(kJ/mol)/K
beta = 1/(T*Kb)

box = 4 #nm

# make target plot
def VLJ(r,epsilon_kcal):
    epsilon_kj = epsilon_kcal*4.184
    sigma = 0.3350 # nanometers
    sigma_o_r = sigma/r
    sigma_o_r6 = (sigma_o_r)**6
    return 4*epsilon_kj*(sigma_o_r6*(sigma_o_r6-1))

def Fanalytic(r,beta,epsilon):
    return -(2*np.log(r)-beta*VLJ(r,epsilon))/beta

# analysis of data in files
def min3(a,b,c):
    if a < b:
        if a < c:
            return a
        else:
            return c
    else:
        if b < c:
            return b
        else:
            return c

def vec_dist(d1,d2,w,j):
    tmp = d1-d2
    for i,t in enumerate(tmp):
        tmp[i] = min3(abs(t),abs(t-box),abs(t+box))

    if np.sqrt(np.sum(np.square(tmp))) > 8:
        print("walker",w,"index",j)
        print("d1 is",d1)
        print("d2 is",d2)
    return np.sqrt(np.sum(np.square(tmp)))

def get_bias_value(d,cycle_idx,k):
    d0 = d0_start + cycle_idx*d0_incr
    return 0.5*k*(d-d0)**2

#def fe_calc(wepy_path, n_bins, file_idx, k=2000):
def fe_calc(file_list, n_bins, k=2000):
    # NUMERATOR
    # for every cycle, do the binning and averaging for all the walkers and sum up for all times

    # This calculates G0 in eq [8] from Hummer, Szabo PNAS 2001 98, 7
    #
    # G0(z) =  -1/beta ln[ (sum_t (term1 / term2)) / ( sum_t (term3 / term2)) ]
    #
    # where
    #
    # term1 =  <delta(z-z_t) exp(-beta w_t) >     # specific to z and t
    # term2 = <exp(-beta w_t)>                    # specific to t
    # term3 = exp[-beta u(z,t)]                   # specific to z and t
    #

    numer = np.zeros((n_bins))
    denom = np.zeros((n_bins))

    g0 = np.zeros((n_bins))
    n_g0 = np.zeros((n_bins))
    d_values = [(i+0.5)*(d_max-d_min)/n_bins for i in range(n_bins)]

    # initialize variables
    term1 = np.zeros((n_bins,n_cycles))
    term2 = np.zeros((n_cycles))
    norm = np.zeros((n_cycles))

    n_part = 2
    n_dim = 3
    n_walkers = walkers

    positions = np.zeros((n_walkers,n_cycles,n_part,n_dim))

    for index,value in enumerate(file_list):

        wepy_h5 = WepyHDF5(value, mode='r')
        wepy_h5.open()

        Zs = np.array(list(wepy_h5.h5['runs/0/resampler/Z'][:,0]))
        Zs_running_prod = [Zs[0:i+1].prod() for i in range(len(Zs))]

        for j in range(n_walkers):
            positions[j] = np.array(wepy_h5.h5['runs/0/trajectories/'+str(j)+'/positions'])

        for cycle in range(n_cycles):
            # these lists have all the distances, work values, and weights for cycle i
            ds_cyc = []

            for j in range(n_walkers):
                # get distances
                p = positions[j,cycle] 
                tmp = vec_dist(p[0], p[1], j, cycle)
                ds_cyc.append(tmp)

            # e_mbwt = np.exp(-beta*np.array(work_cyc))
            # new value is the product of Zs
            e_mbwt = Zs_running_prod[cycle]

            for j,d in enumerate(ds_cyc):
                # find out which bin it's in
                 bin_id = int((d - d_min)/(d_max - d_min)*n_bins)

                 term1[bin_id][cycle] += e_mbwt
                 term2[cycle] += e_mbwt
                 norm[cycle] += 1

            # end of loop over cycles
            # terms have been computed, add to the running sums over timepoints
    for b in range(n_bins):
        numer[b] = 0
        for cycle in range(n_cycles):
            numer[b] += term1[b][cycle]/term2[cycle]

            # Note: get_bias_value returns term3
            # need to use cycle+1 so the d0 matches the work values
            term3 = np.exp(-beta*get_bias_value(d_values[b],cycle+1,k))
            denom[b] += term3/(term2[cycle]/norm[cycle])

        wepy_h5.close()

    g0_no_gaps = []
    d_values_no_gaps = []
    for b in range(n_bins):
        if numer[b] > 0 and denom[b] > 0:
            g0_no_gaps.append(-np.log(numer[b]/denom[b])/beta)
            d_values_no_gaps.append(d_values[b])

    g0_arr = np.array(g0_no_gaps)
    d_val_arr = np.array(d_values_no_gaps)

    plt.plot(d_values_no_gaps,g0_arr-g0_arr.min(), label='FES')
    
    return d_values_no_gaps, g0_no_gaps, g0_arr, d_val_arr

def get_dF(d_values,g0_values,p1,p2):
    dist = np.square(d_values-p1)
    bin1 = np.argmin(dist)

    dist = np.square(d_values-p2)
    bin2 = np.argmin(dist)

    return g0_values[bin2]-g0_values[bin1]
        
if __name__ == "__main__":

    epsilon = int(sys.argv[1])
    walkers = int(sys.argv[2])
    file_name = sys.argv[3]
    
    # make target plot
    p1 = 0.38
    p2 = 1.5
    
    rs = np.linspace(0.3,2,1000)
    Ftarget = Fanalytic(rs,beta,epsilon)
    Ftarget -= Ftarget.min()
    target_dF = Fanalytic(p2,beta,epsilon) - Fanalytic(p1,beta,epsilon)
    
    print('Target plot made')

    # cycle through 10 sets of data
    all_d_val = []
    all_g0_arr = []
    all_dFsq = []
    all_dF = []

    file_list = [file_name]
    d, g, ga, da = fe_calc(file_list, 1000, k=2000)
    all_dFsq.append((get_dF(da,ga,p1,p2) - target_dF)**2)
    all_d_val.append(da)
    all_g0_arr.append(ga)

    plt.plot(rs, Ftarget, color='k', linestyle='--', label='Target')
    plt.xlabel('Interatomic Distance (nm)')
    plt.ylabel('Free Energy (kj/mol)')
    plt.ylim(-2, 100)
    plt.legend()
    plt.show()
    
        
