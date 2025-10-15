import numpy as np
import scipy
from mpi4py import MPI
from Boundary import Boundary
# Class which holds information pertaining to each room in the simulation

class Room():
    # Room needs to know it's dimensions for temperature matrix
    # it needs to know it's boundaries for sparse matrix construction and solving
    # needs to be given the boundaries from the apartment class
    def __init__(self, rank, dim_array, delta_x, comm):
        """
        Give the room the rank of the process it's running on and it's size.
        NOTE: dim_array should be (width, height)
        """
        self.dim = dim_array
        self.scale = int(1/delta_x)
        self.h = delta_x
        self.M = int(dim_array[0] / delta_x)
        self.N = int(dim_array[1] / delta_x)
        # Create a temperature array for this room
        self.u = np.zeros((dim_array / delta_x).astype(np.int16))
        self.V = self.u.flatten()
        # Boundaries will be accessed here for creation of A matrix and B vector
        self.boundaries = []
        self.A = None
        self.B = np.zeros((len(self.V), 1))
        # Slices for boundaries in flattened arrays
        #self.topSlice = slice(0, self.N, 1)
        #self.bottomSlice = slice((self.M - 1)*self.N, self.M*self.N, 1)
        #self.leftSlice = slice(0, (self.M - 1)*self.N + 1, self.N)
        #self.rightSlice = slice(self.N - 1, self.M*self.N, self.N)
        # Indices of boundaries in flattened array (alternative to slicing)
        # Confusing because the array is rotated 90 degrees CW from room setup
        self.leftIndices = np.arange(0, self.N, 1)
        self.rightIndices = np.arange((self.M - 1)*self.N, self.M*self.N, 1)
        self.bottomIndices = np.arange(0, (self.M - 1)*self.N + 1, self.N)
        self.topIndices = np.arange(self.N - 1, self.M*self.N, self.N)
        self.rank = rank
        self.comm = comm
    def get_rank(self):
        return self.rank
    def get_dims(self):
        return self.dim
    def get_scale(self):
        return self.scale
    def get_temp_array(self):
        return self.V.reshape((self.M, self.N))
    def get_boundaries(self):
        return self.boundaries
    def update_V(self, newV, omega):
        # Update the V vector to the newV using relaxation from current with constant omega
        self.V = omega*newV.reshape(1, -1) + (1-omega)*self.V
    
    def recv_sparse_matrix(self, source, shape_tag, data_tag):
        # Receive int length using recv
        data_len = self.comm.recv(source=source, tag=shape_tag)
        data_len = int(data_len)
        mat_data = np.empty((3, data_len))
        self.comm.Recv(mat_data, source=source, tag=data_tag)
        #print(mat_data.shape)
        #print(mat_data)
        # Rebuild matrix, row and col index values must be ints
        A_mat = scipy.sparse.csr_matrix((mat_data[0], (mat_data[1].astype(np.int16), mat_data[2].astype(np.int16))))
        return A_mat
    
    def add_boundaries(self, boundaries_list):
        # Take array of form [[room2,'D', startPos, endPos], [room3, 'N', startPos, endPos]] where room2, etc.. are Room objs
        # and the second value in the tuple is the type of boundary (Dirichlet, Neumann, constant)
        # the third and fourth values are (x, y) start and end positions for the boundary
        # IN ROOM COORDINATES, i.e with (0, 0) at bottom left
        # Also, the endpoints are not inclusive, so you can say (0, 0) to (0, 50) and we interpret
        # that as (0, 0) to (0, 49)
        # Boundary will be taken as slices from start to end not including last
        # If element 0 is a Room object, we have another process on that boundary, otherwise
        # if it is a float it's a constant value
        # Check whether these values are acceptable
        for k in range(len(boundaries_list)):
            bound_arr = boundaries_list[k]
            # Each bound_arr is a list of four elements as above
            bound_neighboor, bound_type, bound_start, bound_end = bound_arr[0:]
            # Check type of boundary
            assert bound_type in ['N', 'D'] or isinstance(bound_type, (int, float)), "Boundary must be either a constant, 'N', or 'D'"
            # Check bounds
            assert bound_end[0]-bound_start[0] == 0 or bound_end[1]-bound_start[1] == 0, "Start and end must be on the same side"
            assert bound_end[0] in [0, self.M] or bound_end[1] in [0, self.N], f"Points must be on boundary {bound_end}, {self.M}, {self.N}"
            assert bound_start[0] in [0, self.M] or bound_start[1] in [0, self.N], f"Points must be on boundary {bound_start}, {self.M}, {self.N}"
            assert np.all(bound_end-bound_start >= 0), "Starting point must have smaller values than ending point"
            
            new_boundary = Boundary(bound_neighboor, bound_type, bound_start, bound_end)
            new_boundary.room_size = self.M, self.N
            self.boundaries.append(new_boundary)
    
    def create_A(self):
        # Create the sparse A matrix
        # We want an array of values, an array of row indices for those values
        # and an array of column indices for those values
        # These can then be fed into the scipy csr sparse matrix
        values = []
        row_indices = []
        col_indices = []
        # Use neighbor average for all internal points
        # v_i,j is ni+j in row ni+j of A
        # Neighbors in A of v_i,j are at positions n(i+1)+j [v_i+1,j], n(i-1)+j [v_i-1,j]
        # ni+j+1 [v_i,j+1], and ni+j-1 [v_i,j-1] in row ni+j
        # Iterate through the points, adding values and row/col indices
        for i in range(0, self.M, 1):
            for j in range(0, self.N, 1):
                # Set up one row of A
                row_ind = self.N*i+j
                # v_ij
                row_indices.append(self.N*i+j)
                col_indices.append(self.N*i+j)
                values.append(-4)
                if i != self.M-1:
                    # v_i+1,j
                    row_indices.append(row_ind)
                    col_indices.append(self.N*(i+1)+j)
                    values.append(1)
                if i != 0:
                    # v_i-1,j
                    row_indices.append(row_ind)
                    col_indices.append(self.N*(i-1)+j)
                    values.append(1)
                if j != self.N-1:
                    # v_i,j+1
                    row_indices.append(row_ind)
                    col_indices.append(self.N*i+j+1)
                    values.append(1)
                if j != 0:
                    # v_i,j-1
                    row_indices.append(row_ind)
                    col_indices.append(self.N*i+j-1)
                    values.append(1)
        # Create matrix A as a compressed sparse row sparse matrix
        A_mat = scipy.sparse.csr_matrix((values, (row_indices, col_indices)))
        # Next look at boundaries
        for bound in self.boundaries:
            # Each bound_arr is a list of four elements as above
            # Get indices along the boundary
            i_inds, j_inds = bound.get_boundary_indices()
            diag_inds = self.N*i_inds+j_inds
            # Boundary condition on A
            if bound.value == 'N':
                # Neumann condition
                # Set the diagonal point equal to -3 instead along this boundary
                # Index into a sparse matrix with array of row inds, array of col inds
                A_mat[diag_inds, diag_inds] = -3
        # Divide by h**2
        A_mat.data = A_mat.data / self.h**2
        self.A = A_mat
        return A_mat
        
    def create_B(self):
        B_vec = np.zeros(self.V.shape)
        # Look at boundaries
        for bound in self.boundaries:
            bound_room = bound.get_neighboor()
            bound_type = bound.get_value()
            # Get indices along the boundary
            i_inds, j_inds = bound.get_boundary_indices()
            # boundary values
            bound_values = np.ones(len(i_inds))
            if bound_type == 'N':
                # Neumann condition
                neighbor_values = self.get_info(self, bound_room)
                bound_values = neighbor_values / self.h
            elif bound_type == 'D':
                # Dirichlet condition
                # Get info and treat as constant boundary 
                neighbor_values = self.get_info(self, bound_room)
                bound_values = neighbor_values / self.h**2
            else:
                # Constant given in bound_type
                bound_values *= bound_type / self.h**2
            # We now have the boundary values to subtract from current values
            for i in range(len(i_inds)):
                B_vec[self.N*i_inds[i]+j_inds[i]] -= bound_values[i]
        self.B = B_vec
        return B_vec
                
    # Function to get information from another room
    def get_info(self, current_room, destination_room):
        # Should be able to check whether destination room has a border with current room
        # If so, destination room knows how many values to send back to current room by
        # checking its own boundary conditions for the one specifying current room
        dest_boundaries = destination_room.get_boundaries()
        #dest_room_rank = destination_room.get_rank()
        found_room = False
        for bound in dest_boundaries:
            # Each bound_arr is a list of four elements as above
            bound_room = bound.get_neighboor()
            if bound_room != current_room:
                continue
            # else
            found_room = True
            # Indices of the points in the room specified in this boundary condition
            # Get indices along the boundary
            i_inds, j_inds = bound.get_boundary_indices()
            # Send this information back to the asking room as a vector
            
            info = np.zeros(len(i_inds))
            for i in range(len(i_inds)):
                info[i] = destination_room.V[destination_room.N*i_inds[i]+j_inds[i]]
            return info
        if not found_room:
            raise ValueError(f"Destination room {destination_room.get_rank()} shares no boundary with current room {current_room.get_rank()}")
           
    
    # Solve method
    def solve(self, iterations, omega):
        # Perform an update step on the room
        # Create new boundary vector, A should be the same
        rank = self.comm.Get_rank()
        #print(rank)
        for it in range(iterations):
            #print(f"Starting solve in room {rank}")
            self.comm.Recv(self.B, source=0, tag=rank*100 + it + 1)
            #print(self.B.shape)
            #print(f"Recieved B: {self.B} in room {self.comm.Get_rank()}")
            new_V = scipy.sparse.linalg.spsolve(self.A, self.B)
            #print(f"Computed flat vector: {new_V.shape}")
            # Relaxation
            #print(f"V vector: {self.V}")
            self.update_V(new_V, omega)
            #print(f"{new_V.shape}, {self.V.shape}")
            #print(f"from room: {self.V.shape}")
            #print(f"New v: {self.V}")
            self.V = self.V.flatten()
            self.comm.Send([self.V, MPI.DOUBLE], dest=0, tag=rank*1000 + it + 1)

if __name__ == "__main__":
    vec = np.ones(4).reshape(4, 1)
    omega = 0.5
    room = Room(0, np.array([1, 1]), 1/2, None)
    room.update_V(vec, omega)
    print(room.V)
