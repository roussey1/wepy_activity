import numpy as np

class OnePropertyDistance(object):

    def __init__(self, prop):
        self.prop = prop
        
    def image(self, state):
        return state[self.prop]

    def image_distance(self, image_a, image_b):
        return np.abs(image_a - image_b)

    def distance(self, state_a, state_b):
        """ Compute the distance between two states. """

        return self.image_distance(self.image(state_a),
                                      self.image(state_b))

