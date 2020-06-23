class Activity(object):

    def __init__(self):
        pass

    def get_incr(self, oldstate, newstate):
        """Returns the activity increment of a trajectory segment given the
        initial (oldstate) and final (newstate) simulation states.
        """
        raise NotImplementedError
