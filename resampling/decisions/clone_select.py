from collections import namedtuple, defaultdict
from enum import Enum
import logging

import numpy as np

from wepy.resampling.decisions.decision import Decision

from wepy.walker import split, keep_merge

# the possible types of decisions that can be made enumerated for
# storage, these each correspond to specific instruction type
class CloneSelectDecisionEnum(Enum):
    """Enum definition for cloning and selection decision values."

    - CLONE : 1
    - SELECT : 2
    - SQUASH : 3

    """

    CLONE = 1
    """Clone the walker into multiple children equally splitting the weight."""

    SELECT = 2
    """Walker is given an equal weight to other selected walkers."""

    SQUASH = 3
    """Destroy the walker sample value (state)."""
    

class CloneSelectDecision(Decision):
    """Decision encoding cloning and merging decisions for weighted ensemble.

    The decision records have in addition to the 'decision_id' a field
    called 'target_idxs'. This field has differing interpretations
    depending on the 'decision_id'.

    For NOTHING and KEEP_MERGE it indicates the walker index to assign
    this sample to after resampling. In this sense the walker is
    merely a vessel for the propagation of the state and acts as a
    slot.

    For SQUASH it indicates the walker that it's weight will be given
    to, which must have a KEEP_MERGE record for it.

    For CLONE it indicates the walker indices that clones of this one
    will be placed in. This field is variable length and the length
    corresponds to the number of clones.

    """


    ENUM = CloneSelectDecisionEnum

    DEFAULT_DECISION = ENUM.SQUASH

#    print(Decision.FIELDS)
    FIELDS = Decision.FIELDS + ('target_idxs',) #('target_idxs',)
    SHAPES = Decision.SHAPES + (Ellipsis,)
    DTYPES = Decision.DTYPES + (np.int,)

    RECORD_FIELDS = Decision.RECORD_FIELDS + ('target_idxs',)

    # the decision types that pass on their state
    ANCESTOR_DECISION_IDS = (ENUM.CLONE.value,
                             ENUM.SELECT.value)

    # TODO deprecate in favor of Decision implementation
    @classmethod
    def record(cls, enum_value, target_idxs):
        record = super().record(enum_value)
        record['target_idxs'] = target_idxs

        return record

    @classmethod
    def action(cls, walkers, decisions):

        # list for the modified walkers
        mod_walkers = [None for i in range(len(walkers))]

        # perform clones and merges for each step of resampling
        for step_idx, step_recs in enumerate(decisions):
            
            # determine the number of selected walkers
            n_select = 0
            for rec in step_recs:
                if rec['decision_id'] == cls.ENUM.SELECT.value:
                    n_select += len(rec['target_idxs'])

            # determine the "left-over" weight to be divided evenly
            # amongst the selected walkers
            #
            # this will be equal to the sum of the weights of all
            # walkers that were not cloned
            w_leftover = 0
            for walker_idx, rec in enumerate(step_recs):
                if rec['decision_id'] != cls.ENUM.CLONE.value:
                    w_leftover += walkers[walker_idx].weight

            # set w_select to be the leftover weight divided by
            # the number of selected walkers
            w_select = w_leftover / n_select
            
            # go through each decision and perform the decision
            # instructions
            for walker_idx, walker_rec in enumerate(step_recs):

                decision_value = walker_rec['decision_id']
                instruction = walker_rec['target_idxs']

                if decision_value == cls.ENUM.CLONE.value:

                    # get the walker to be cloned
                    walker = walkers[walker_idx]
                    # "clone" it by splitting it into walkers of the
                    # same state with even weights
                    clones = split(walker, number=len(instruction))

                    # then assign each of these clones to a target
                    # walker index in the next step
                    for clone_idx, target_idx in enumerate(instruction):

                        # check that there is not another walker
                        # already assigned to this position
                        if mod_walkers[target_idx] is not None:
                            raise ValueError(
                                "Multiple walkers assigned to position {}".format(instruction[0]))

                        # assign the clone to the modified walkers of the next step
                        mod_walkers[target_idx] = clones[clone_idx]

                # if it is a decision for merging we must perform this
                # once we know all the merge targets for each merge group
                elif decision_value == cls.ENUM.SQUASH.value:
                    # Do nothing
                    pass

                elif decision_value == cls.ENUM.SELECT.value:
                    # walker was selected for survival.  congratulations!
                    walker = walkers[walker_idx]
                    for tgt in instruction:
                        mod_walkers[tgt] = type(walker)(walker.state,w_select)
                        
                else:
                    raise ValueError("Decision not recognized")

        if None in mod_walkers:
            raise ValueError("Some walkers were not created")

        return mod_walkers
