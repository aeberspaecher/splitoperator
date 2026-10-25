"""Defines and implements the recorder object used to record split-operator
time-propagation.
"""


class Recorder:
    """Record data from split operator time propagation. Also save all relevant
    data.
    """

    # relevant data:
    # - psi0
    # - grid dimensions
    # - potential (V(t))
    # - psi(t)


    def __init__(self, grid, psi0, V, absorber=None, V_is_time_dep=False):
        pass

    def add_iterate(self, psi, t):
        pass

    def save_to_h5(self, filename):
        pass


class RecorderFromData(Recorder):

    def __init__(self, filename):
        """Create a Recorder from data stored in an h5 file.

        This class may be used to load calculations from a file for either
        continuing the calculation or feeding the analyzers with precomputed
        data.
        """

        pass
