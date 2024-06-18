import numpy as np
import misc
class Force:
    def __init__(self, F):
        if not callable(F):
            self.F=lambda _:misc.np_ndarray(F)
        else:
            self.F=F

    def update(self, obj: object):
        return self.F(obj)
