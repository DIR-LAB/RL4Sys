class RL4SysObservation:
    def __init__(self, data):
        """
        Initialize an RL4SysObservation instance.

        Args:
            data: The data representing the state or observation. The type and structure
                  of this data will depend on the specific application. It could be a
                  simple scalar value, a vector, a matrix, or even more complex data structures.
        """
        self.data = data

    # You can add additional methods here for manipulating or interpreting
    # the observation data, such as normalization, encoding, etc.
