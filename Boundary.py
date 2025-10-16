import numpy as np

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

        self.room_size = None

    def get_value(self):
        return self.value

    def get_neighboor(self):
        return self.neighbor

    def get_data(self):
        return self.value, self.start, self.end 

    def get_boundary_indices(self):
        """
            Get grid indices that are contained in this boundary segment
        """
        # Indices of the points in the room specified in this boundary condition
        i_inds = np.arange(self.start[0], self.end[0])
        j_inds = np.arange(self.start[1], self.end[1])
        # One of i_inds and j_inds will be empty, replace it with the constant index of the right length
        if len(i_inds) == 0:
            # If starting bound in x is on the edge, adjust so no indexing error
            if self.start[0] == self.room_size[0]:
                i_inds = np.ones(len(j_inds), dtype=np.int16) * (self.start[0]-1)
            else:
                i_inds = np.ones(len(j_inds), dtype=np.int16) * self.start[0]
        elif len(j_inds) == 0:
            # If starting bound in y is on the edge, adjust so no indexing error
            if self.start[1] == self.room_size[1]:
                j_inds = np.ones(len(i_inds), dtype=np.int16) * (self.start[1]-1)
            else:
                j_inds = np.ones(len(i_inds), dtype=np.int16) * self.start[1]
        else:
            print("PROBLEM: i_inds or j_inds should have been empty")
        
        return i_inds, j_inds
    
    def get_plotting_indices(self):
        """
            Get grid indices that are contained in this boundary segment, allowing for starting out of bounds since
            that reflects how the plotting will look. Also include end point for the plot.
        """
        # Linspace from start to end inclusive
        diff = self.end[0]-self.start[0]
        num_pts = diff + 1
        if diff == 0:
            num_pts = self.end[1]-self.start[1]+1
        i_inds = np.linspace(self.start[0], self.end[0], num_pts)
        j_inds = np.linspace(self.start[1], self.end[1], num_pts)
        
        return i_inds, j_inds
    