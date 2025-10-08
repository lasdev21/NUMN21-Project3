import numpy as np
# Class which holds information pertaining to each room in the simulation

class Room():
    # Room needs to know it's dimensions for temperature matrix
    # it needs to know it's boundaries for sparse matrix construction and solving
    # needs to be given the boundaries from the apartment class
    def __init__(self, process_rank, dim_array, delta_x):
        """
        Give the room the rank of the process it's running on and it's size.
        NOTE: dim_array should be (width, height)
        """
        self.rank = process_rank
        self.dim = dim_array
        self.scale = int(1/delta_x)
        # Create a temperature array for this room
        self.u = np.zeros((dim_array / delta_x).astype(np.int16))
    def get_dims(self):
        return self.dim
    def get_scale(self):
        return self.scale
    def get_temp_array(self):
        return self.u
    
    def add_boundaries(self, boundary_array):
        # Take array of form [[room2,'D', 'top'], [room3, 'N', 'left']] where room2, etc.. are Room objs
        # and the second value in the tuple is the type of boundary (Dirichlet, Neumann, constant)
        # the third value is the wall on which the boundary is found
        # If element 0 is a Room object, we have another process on that boundary, otherwise
        # if it is a float it's a constant value
        
        pass
    
    # Solve method
    def solve(self):
        # Could optionally take the ranks of rooms bordering it.
        pass
