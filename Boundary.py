import numpy as np

def dirilchlet_condition():
    pass

class Boundary():
    def __init__(self, neighbor, value, location_start, location_end):
        """
        neighbor:   Room to which the boundary is connected. None if it is not connected to any room.
        value   :   Dirichlet ('D') or Neumann ('N') or constant number
        location:   2 points that represents the limits of the (left to right or top to bottom)
        """
        self.neighbor = neighbor
        self.value = value
        self.start = location_start
        self.end = location_end

    def get_data(self):
        return self.value, self.start, self.end 

    def get_boundary_indices(self, room_size):
        """
            Get grid indices that are contained in this boundary segment
        """
        
        pass
    