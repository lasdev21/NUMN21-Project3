import numpy as np

class Boundary():
    def __init__(self, neighbor, value, location_start, location_end, room_size):
        """
        neighbor:   Room to which the boundary is connected. None if it is not connected to any room.
        value   :   Dirichlet ('D') or Neumann ('N') or constant number
        location:   2 points that represents the limits of the (left to right or top to bottom)
        """
        self.neighbor = neighbor
        self.value = value
        self.start = location_start
        self.end = location_end

        self.room_size = room_size
        
        # Boundary indices
        self.generate_boundary_indices()
        #print(f"Boundary\nx: {self.i_inds}\ny:{self.j_inds}\nInner x: {self.inner_i_inds}\nInner y: {self.inner_j_inds}")

    def get_value(self):
        return self.value

    def get_neighbor(self):
        return self.neighbor

    def get_data(self):
        return self.value, self.start, self.end
    
    def get_boundary_indices(self):
        return self.i_inds, self.j_inds
    def get_inner_boundary_indices(self):
        return self.inner_i_inds, self.inner_j_inds

    def generate_boundary_indices(self):
        """
            Generate grid indices that are contained in this boundary segment
        """
        # Indices of the points in the room specified in this boundary condition
        i_inds = np.arange(self.start[0], self.end[0])
        j_inds = np.arange(self.start[1], self.end[1])
        # Indices of the inner boundary
        # Start as copies, one will remain a copy and the other will be incremented/decremented by 1
        inner_i_inds = i_inds.copy()
        inner_j_inds = j_inds.copy()
        # One of i_inds and j_inds will be empty, replace it with the constant index of the right length
        if len(i_inds) == 0:
            # If starting bound in x is on the edge, adjust so no indexing error
            if self.start[0] == self.room_size[0]: # On right, i_inds are all M-1
                i_inds = np.ones(len(j_inds), dtype=np.int64) * (self.start[0]-1)
                # Inner inds are all M-2 then
                inner_i_inds = i_inds - 1
            else: # On left, i_inds are all 0
                i_inds = np.ones(len(j_inds), dtype=np.int64) * self.start[0]
                # Inner inds are all 1 then
                inner_i_inds = i_inds + 1
        elif len(j_inds) == 0:
            # If starting bound in y is on the edge, adjust so no indexing error
            if self.start[1] == self.room_size[1]: # On top, j_inds are all N-1
                j_inds = np.ones(len(i_inds), dtype=np.int64) * (self.start[1]-1)
                # Inner inds are all N-2 then
                inner_j_inds = j_inds - 1
            else: # On bottom, j_inds are all 0
                j_inds = np.ones(len(i_inds), dtype=np.int64) * self.start[1]
                # Inner inds are all 1 then
                inner_j_inds = j_inds + 1
        else:
            print("PROBLEM: i_inds or j_inds should have been empty")
        
        # Set
        self.i_inds = i_inds
        self.j_inds = j_inds
        self.inner_i_inds = inner_i_inds
        self.inner_j_inds = inner_j_inds
        
    
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
    