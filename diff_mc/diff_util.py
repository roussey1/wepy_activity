import numpy as np

### Kres using int was doing 3.99999999 = 3, not 4
### This was causing sum(n) to not equal the number of walkers in the ensemble (len(weights))
### Using np.rint() allows rounding to the nearest whole number, so 3.9999 = 4.0
### This makes it so that len(n) = len(weights)
### and stops the assert in line 118 in the resampler

### originally thought that could fix with regenerating n in a loop until
### len(n) = len(weights) but was purely a rounding issue


# def get_branch_nums(weights):
#     """
#     This function follows the algorithm on Page 416 (Section 6.1.2.3)
#     of Free Energy Computations: A Mathematical Perspective by 
#     Rousset et al.
#     """
#     # get number of walkers 
#     K = len(weights)
    
#     # get residual weights
#     r = K*weights - np.floor(K*weights)

#     m = _generate_m(K, weights, r)

#     # return m
#     print(m, np.sum(m), float(len(weights)))

#     while np.sum(m) != len(weights):
#         m = _generate_m(K, weights, r)
#         print(m)

#     return m


# def _generate_m(K, weights, r):

#     Kres = np.rint(np.array(np.sum(r)))
#     choices = np.random.choice(range(K),size=(int(Kres)),p=r/r.sum()) 
 
#     m = np.floor(K*weights)
#     for c in choices:
#         m[c] += 1
        
#     return m


def get_branch_nums(weights):
    """
    This function follows the algorithm on Page 416 (Section 6.1.2.3)
    of Free Energy Computations: A Mathematical Perspective by 
    Rousset et al.
    """
    # get number of walkers 
    K = len(weights)
    
    # get residual weights
    r = K*weights - np.floor(K*weights)

    # get the sum 
    Kres = np.rint(np.sum(r))

    # choose Kres numbers with weights corresponding to r
    choices = np.random.choice(range(K),size=(int(Kres)),p=r/r.sum()) 

    n = np.floor(K*weights)
    for c in choices:
        n[c] += 1
        
    return n

def get_entropy(weights):
    """
    This function gets the relative entropy, as described in
    Section 6.1.2.2 of Rousset.
    """
    
    return np.sum(weights*np.log(len(weights)*weights))
