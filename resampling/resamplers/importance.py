import multiprocessing as mulproc
import random as rand
import itertools as it

import numpy as np

from wepy.resampling.resamplers.resampler import Resampler
from resampling.decisions.clone_select import CloneSelectDecision
from wepy.resampling.resamplers.clone_merge  import CloneMergeResampler

class Importance(object):
    def __init__(self):
        pass

    def calcImportance(self, walker):
        raise NotImplementedError

class JarzynskiImportance(Importance):
    def __init__(self, beta=None, amplification=1):

        assert beta is not None, "Must define inverse temperature (beta)"
        self.beta = beta
        self.amplification = amplification

    def calcImportance(self, walker):
        return walker.weight*np.exp(-self.beta*self.amplification*walker.state['activity'][0][0])

class ImportanceResampler(Resampler):
    """
    The Importance resampler uses an importance function value for each walker
    I(w) to decide which trajectories to clone and merge.  A subset of the N 
    trajectories are chosen for survival strictly according to the importance function, 
    with the remaining walkers being selected according to their weight.

    If walkers are chosen more than once, they are said to be cloned and 
    their weight is distributed among the resulting walkers.
    
    For the number of walkers selec

    """
    DECISION = CloneSelectDecision

    # state change data for the resampler
    RESAMPLER_FIELDS = ('n_walkers', 'importance')
    RESAMPLER_SHAPES = ((1,), Ellipsis)
    RESAMPLER_DTYPES = (np.int, np.float)

    # fields for resampling data
#    import ipdb; ipdb.set_trace()
    RESAMPLING_FIELDS = DECISION.FIELDS + ('step_idx', 'walker_idx',)
    RESAMPLING_SHAPES = DECISION.SHAPES + ((1,), (1,),)
    RESAMPLING_DTYPES = DECISION.DTYPES + (np.int, np.int,)

    # fields that can be used for a table like representation
    RESAMPLING_RECORD_FIELDS = DECISION.RECORD_FIELDS + ('step_idx', 'walker_idx',)

    def __init__(self, seed=None, imp=None, n_wt_based=1):

        #import ipdb; ipdb.set_trace()
        #self.decision = self.DECISION

        # the importance
        assert isinstance(imp,Importance), "Must give an Importance object"
        self.imp = imp

        # setting the random seed
        self.seed = seed
        if seed is not None:
            rand.seed(seed)

        self.n_wt_based = n_wt_based

    # we need this to on the fly find out what the datatype of the
    # image is
    # def resampler_field_dtypes(self):

    #     # index of the image idx
    #     import ipdb; ipdb.set_trace()
    #     image_idx = self.resampler_field_names().index('images')

    #     # dtypes adding the image dtype
    #     dtypes = list(super().resampler_field_dtypes())
    #     dtypes[image_idx] = self.image_dtype

    #     return tuple(dtypes)

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


    def decide_clone_merge(self, walkerwt, imp_vals):

        n_walkers = len(walkerwt)

        n_by_importance = n_walkers - self.n_wt_based

        imp_sum = imp_vals.sum()

        walker_clone_nums = np.zeros((n_walkers))
        
        # select n_by_importance walkers by their importance values
        for i in range(n_by_importance):
            # choose a random walker with a probability proportional to its importance
            clone_idx = np.random.choice(range(n_walkers),p=imp_vals/imp_sum)
            walker_clone_nums[clone_idx] += 1

        # select n_wt_based walkers by their weights

        walker_select_nums = np.zeros((n_walkers))
        
        wt_no_clones = walkerwt[walker_clone_nums == 0]
        wt_sum_no_clones = wt_no_clones.sum()
        walkers_no_clones = np.array(range(n_walkers))[walker_clone_nums == 0]

        for i in range(self.n_wt_based):
            # choose a random walker with a probability proportional to its weight
            select_idx = np.random.choice(walkers_no_clones,p=wt_no_clones/wt_sum_no_clones)
            walker_select_nums[select_idx] += 1

        # initiate walker_actions and bypass assign_clones function
        # defautl decision '3', squash
        walker_actions = self._init_walker_actions(n_walkers)
        
        # create records for cloned walkers
        tgt_w = 0
        for walker_idx in range(n_walkers):
            if walker_clone_nums[walker_idx] > 0:
                #build list of targets
                target_idxs = []
                for i in range(int(walker_clone_nums[walker_idx])):
                    target_idxs.append(tgt_w)
                    tgt_w += 1

                #create record 
                walker_actions[walker_idx] = self.decision.record(self.decision.ENUM.CLONE.value,
                                                                  target_idxs=tuple(target_idxs))

        #import ipdb; ipdb.set_trace()
        assert tgt_w == n_by_importance, "Number of cloned walkers is incorrect"

        # create records for selected walkers (starting with tgt_w = n_by_importance)
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

        # calculate importance values
        imp_vals = np.array([self.imp.calcImportance(w) for w in walkers])

        if debug_prints:
            print("Importance values:")
            print(imp_vals)

        # determine cloning and merging actions to be performed
        resampling_data = self.decide_clone_merge(walkerwt, imp_vals)

        # convert the target idxs and decision_id to feature vector arrays
        for record in resampling_data:
            record['target_idxs'] = np.array(record['target_idxs'])
            record['decision_id'] = np.array([record['decision_id']])

        # actually do the cloning and merging of the walkers
        resampled_walkers = self.decision.action(walkers, [resampling_data])

        # flatten the distance matrix and give the number of walkers
        # as well for the resampler data, there is just one per cycle
        resampler_data = [{'importance' : imp_vals,
                           'n_walkers' : np.array([len(walkers)])}]

        return resampled_walkers, resampling_data, resampler_data

