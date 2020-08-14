import multiprocessing as mulproc
import random as rand
import itertools as it

import numpy as np

from wepy.resampling.resamplers.resampler import Resampler
from resampling.decisions.clone_select import CloneSelectDecision
from wepy.resampling.resamplers.clone_merge  import CloneMergeResampler
from diff_mc.diff_util import get_branch_nums, get_entropy

class Importance(object):
    def __init__(self):
        pass

    def calcImportance(self, walker):
        raise NotImplementedError

class DiffMCImportance(Importance):
    def __init__(self, beta=None):

        assert beta is not None, "Must define inverse temperature (beta)"
        self.beta = beta

    def calcImportance(self, walker):
        act = walker.state['activity'][0][0]
        return self._calcImportance(act)

    def _calcImportance(self, activity):
        return np.exp(-self.beta*activity)

class DiffMCResampler(Resampler):
    """
    """
    DECISION = CloneSelectDecision

    # state change data for the resampler
    RESAMPLER_FIELDS = ('n_walkers', 'diffmc_wts','Z')
    RESAMPLER_SHAPES = ((1,), Ellipsis,(1,))
    RESAMPLER_DTYPES = (np.int, np.float, np.float)

    # fields for resampling data
    RESAMPLING_FIELDS = DECISION.FIELDS + ('step_idx', 'walker_idx',)
    RESAMPLING_SHAPES = DECISION.SHAPES + ((1,), (1,),)
    RESAMPLING_DTYPES = DECISION.DTYPES + (np.int, np.int,)

    # fields that can be used for a table like representation
    RESAMPLING_RECORD_FIELDS = DECISION.RECORD_FIELDS + ('step_idx', 'walker_idx',)

    def __init__(self, seed=None, imp=None, ent_cutoff=None):
        
        #self.decision = self.DECISION

        # the importance
        assert isinstance(imp,Importance), "Must give an Importance object"
        self.imp = imp
        self.ent_cutoff = ent_cutoff
        self.Z = [1.0]

        # setting the random seed
        self.seed = seed
        if seed is not None:
            rand.seed(seed)

    def _init_walker_actions(self, n_walkers):
        """Returns a list of default resampling records for a single
        resampling step.

        Parameters
        ----------

        n_walkers : int
            The number of walkers to generate records for

        Returns
        -------

        decision_records : list of dict of str: value
            A list of default decision records for one step of
            resampling.

        """

        # determine resampling actions
        walker_actions = [self.decision.record(
                                enum_value=self.decision.default_decision().value,
                                target_idxs=(i,))
                    for i in range(n_walkers)]

        return walker_actions


    def decide_clone_merge(self, diff_mc_wts):

        n_walkers = len(diff_mc_wts)

        # get the walker branch numbers
        walker_select_nums = get_branch_nums(diff_mc_wts)

        # initiate walker_actions and bypass assign_clones function
        # default decision '3', squash
        walker_actions = self._init_walker_actions(n_walkers)
        
        # create records for selected walkers 
        tgt_w = 0
        for walker_idx in range(n_walkers):
            if walker_select_nums[walker_idx] > 0:

                # build list of targets
                target_idxs = []
                for i in range(int(walker_select_nums[walker_idx])):
                    target_idxs.append(tgt_w)
                    tgt_w += 1

                # create record 
                walker_actions[walker_idx] = self.decision.record(self.decision.ENUM.SELECT.value,
                                                                  target_idxs=tuple(target_idxs))
        assert tgt_w == n_walkers, "Number of selected walkers is incorrect"

        # because there is only one step in resampling here we just
        # add another field for the step as 0 and add the walker index
        # to its record as well
        for walker_idx, walker_record in enumerate(walker_actions):
            walker_record['step_idx'] = np.array([0])
            walker_record['walker_idx'] = np.array([walker_idx])

        return walker_actions

    def resample(self, walkers, debug_prints=False):

        n_walkers = len(walkers)
        walkerwt = np.array([walker.weight for walker in walkers])

        # calculate initial importance values
        diffmc_wts = np.array([self.imp.calcImportance(w) for w in walkers])
        new_Z = np.mean(diffmc_wts)
        
        # normalize diffmc_wts
        diffmc_wts /= np.sum(diffmc_wts)

        # check weight entropy
        if get_entropy(diffmc_wts) > self.ent_cutoff:
            # calculate activities
            activities = np.array([w.state['activity'][0][0] for w in walkers])

            if debug_prints:
                print("Diff MC weights:")
                print(diffmc_wts)

            # determine cloning and merging actions to be performed
            resampling_data = self.decide_clone_merge(diffmc_wts)

            # convert the target idxs and decision_id to feature vector arrays
            for record in resampling_data:
                record['target_idxs'] = np.array(record['target_idxs'])
                record['decision_id'] = np.array([record['decision_id']])

            # actually do the cloning and merging of the walkers
            resampled_walkers = self.decision.action(walkers, [resampling_data])

            # reset the activites
            for w in resampled_walkers:
                w.state['activity'][0][0] = 0

            # keep track of weight normalization factor
            self.Z.append(new_Z)

            # flatten the distance matrix and give the number of walkers
            # as well for the resampler data, there is just one per cycle
            resampler_data = [{'diffmc_wts' : diffmc_wts,
                               'n_walkers' : np.array([len(walkers)]),
                               'Z' : np.array([self.Z[-1]])}]
        else:
            
            last_val = self.Z[-1]
            self.Z.append(last_val)

            walker_actions = self._init_walker_actions(n_walkers)

            for walker_idx in range(len(walkers)):
                # create record 
                walker_actions[walker_idx] = self.decision.record(self.decision.ENUM.SELECT.value,
                                                                  target_idxs=(walker_idx))

            # correct the walker action format to work with the hdf5 reporter
            ids = [] 
            for i in range(n_walkers): 
                ids.append(walker_actions[i]['decision_id']) 

            for index,value in enumerate(ids):
                walker_actions[index].update({'decision_id': np.array([value])})
                
            for i in range(n_walkers):
                walker_actions[i].update({'target_idxs': np.array([i])})
                walker_actions[i].update({'step_idx': np.array([0])})
                walker_actions[i].update({'walker_idx': np.array([i])})


            resampled_walkers = walkers
            resampling_data = walker_actions
            resampler_data = [{'diffmc_wts' : diffmc_wts,
                               'n_walkers' : np.array([len(walkers)]),
                               'Z' : np.array([self.Z[-1]])}]

        return resampled_walkers, resampling_data, resampler_data

