# Class for the apartment, which knows about it's rooms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Room import Room
from mpi4py import MPI

class Apartment():
    # Know about its rooms, how many, how they are connected
    # Needs to be told where the rooms are and connect them
    # NOTE: In the floor plan array, the apartment is represented so that you
    # can build the rooms using (x, y) values as you might graph on a paper. This means
    # it actually is kind of horizontal in the array, but should work.
    def __init__(self, comm):
        self.rooms = []
        self.room_locs = []
        # Floor plan contains an array of values representing the apartment
        # Rooms are then inserted at specified locations in the floor plan
        # and take up the specified size and shape in the array
        # NOTE: (x, y) are measured from bottom left of a room
        # so if I add a room at (2, 1) it adds the bottom left of the room at (2, 1)
        self.floor_plan = np.array([[-1]])
        self.comm = comm
    
    def send_sparse_matrix(self, matrix, dest, shape_tag, data_tag):
        # Convert to a COO sparse matrix so data is all same shape
        coo_mat = matrix.tocoo()
        # Extract the sparse matrix data
        data = coo_mat.data
        row_indices = coo_mat.row
        col_indices = coo_mat.col
        assert len(data) == len(row_indices) == len(col_indices), "All arrays should be same length"
        data_len = len(data)
        # Send the length to the room for creation of empty recv obj
        self.comm.send(data_len, dest=dest, tag=shape_tag)
        # Send the data itself as a vstack
        mat_data = np.vstack((data, row_indices, col_indices))
        self.comm.Send([mat_data, MPI.DOUBLE], dest=dest, tag=data_tag)
    
    # Function to create an apartment that matches the design in project 3
    def initialize_apartment_proj3(self, delta_x):
        # Create the apartment in project 3, containing one room 1x1 connected at the lower left
        # of a larger room 2x1, with another 1x1 room connected on the top right.
        # Add room of size 1x1 at origin
        room1 = Room(1, np.array([1, 1]), delta_x, None)
        
        room2 = Room(2, np.array([1, 2]), delta_x, None)
        
        room3 = Room(3, np.array([1, 1]), delta_x, None)

        #print("Initial plan:")
        #print(self)
        #print()
        # Add room 1 at (0, 0)
        room1_loc = np.array([0, 0])
        self.room_locs.append(room1_loc)
        self.add_room_to_plan(room1, room1_loc)
        self.rooms.append(room1)
        #print("Room 1 added:")
        #print(self)
        #print()
        # Add room 2 at (1, 0)
        room2_loc = np.array([1, 0])
        self.room_locs.append(room2_loc)
        self.add_room_to_plan(room2, room2_loc)
        self.rooms.append(room2)
        #print("Room 2 added:")
        #print(self)
        #print()
        # Add room 3 at (2, 1)
        room3_loc = np.array([2, 1])
        self.room_locs.append(room3_loc)
        self.add_room_to_plan(room3, room3_loc)
        self.rooms.append(room3)
        #print("Room 3 added:")
        #print(self)
        #print()
        
        # Add boundaries
        # Room 1 has top and bottom constant 15, left constant 40, right Neumann with Room2
        room1_scale = room1.get_scale()
        # top, left, bottom, right ordering, not that it matters
        room1_boundaries = [[None, 15, np.array([0, room1_scale]), np.array([room1_scale, room1_scale])],
                            [None, 40, np.array([0, 0]), np.array([0, room1_scale])],
                            [None, 15, np.array([0, 0]), np.array([room1_scale, 0])],
                            [room2, 'N', np.array([room1_scale, 0]), np.array([room1_scale, room1_scale])]]
        room1.add_boundaries(room1_boundaries)
        room1.create_A()
        #print(f"Sparse and normal matrices are equal: {np.all(A_norm == A_sparse.toarray())}")
        self.send_sparse_matrix(room1.A, dest=1, shape_tag=0, data_tag=1)
        
        #print(room1.A)
        # Room 2 has top constant 40, bottom constant 5, left dirichlet with room1 on bottom half and constant 15
        # on top half of left, right constant 15 on bottom half and dirichlet with room3 on top half of right
        room2_scale = room2.get_scale()
        room2_dims = room2.get_dims() # array s.t. scale * dims gives coords of top right in (x, y)
        r2_size = room2_scale*room2_dims
        # top, left, bottom, right ordering, not that it matters
        room2_boundaries = [[None, 40, np.array([0, r2_size[1]]), np.array([r2_size[0], r2_size[1]])],
                            [room1, 'D', np.array([0, 0]), np.array([0, r2_size[1]//2])], # left bottom half
                            [None, 15, np.array([0, r2_size[1]//2]), np.array([0, r2_size[1]])], # left top half
                            [None, 5, np.array([0, 0]), np.array([r2_size[0], 0])],
                            [None, 15, np.array([r2_size[0], 0]), np.array([r2_size[0], r2_size[1]//2])], # right bottom half
                            [room3, 'D', np.array([r2_size[0], r2_size[1]//2]), np.array([r2_size[0], r2_size[1]])]] # right top half
        room2.add_boundaries(room2_boundaries)
        room2.create_A()
        self.send_sparse_matrix(room2.A, dest=2, shape_tag=2, data_tag=3)
        #print(room1.A)
        #room2.V[0:2] = 100 # Can set values in room2 and see the boundary condition updated in room1!
        # Room 3 has top and bottom constant 15, left Neumann with Room 2, right constant 40
        room3_scale = room3.get_scale()
        # top, left, bottom, right ordering, not that it matters
        room3_boundaries = [[None, 15, np.array([0, room3_scale]), np.array([room3_scale, room3_scale])],
                            [room2, 'N', np.array([0, 0]), np.array([0, room3_scale])],
                            [None, 15, np.array([0, 0]), np.array([room3_scale, 0])],
                            [None, 40, np.array([room3_scale, 0]), np.array([room3_scale, room3_scale])]]
        room3.add_boundaries(room3_boundaries)
        room3.create_A()
        self.send_sparse_matrix(room3.A, dest=3, shape_tag=4, data_tag=5)
        #print(room3.A)
        
    # Function to create an apartment that matches the design in project 3a extension
    def initialize_apartment_proj3a(self, delta_x):
        # Create the apartment in project 3 extension, containing one room 1x1 connected at the lower left
        # of a larger room 2x1, with another 1x1 room connected on the top right and a 0.25x0.25 room
        # connected just below the top right room.
        
        # Create rooms of the right sizes
        # scale by 2 so the smallest room can be 1x1
        room1 = Room(1, np.array([1, 1])*2, delta_x, None) 
        room2 = Room(2, np.array([1, 2])*2, delta_x, None)
        room3 = Room(3, np.array([1, 1])*2, delta_x, None)
        room4 = Room(4, np.array([1, 1]), delta_x, None)
        
        # Add room 1 at (0, 0)
        room1_loc = np.array([0, 0])
        self.room_locs.append(room1_loc)
        self.add_room_to_plan(room1, room1_loc)
        self.rooms.append(room1)
        
        # Add room 2 at (2, 0)
        room2_loc = np.array([2, 0])
        self.room_locs.append(room2_loc)
        self.add_room_to_plan(room2, room2_loc)
        self.rooms.append(room2)
        
        # Add room 3 at (4, 2)
        room3_loc = np.array([4, 2])
        self.room_locs.append(room3_loc)
        self.add_room_to_plan(room3, room3_loc)
        self.rooms.append(room3)
        
        # Add room 4 at (4, 1)
        room4_loc = np.array([4, 1])
        self.room_locs.append(room4_loc)
        self.add_room_to_plan(room4, room4_loc)
        self.rooms.append(room4)
        #print("Room 4 added:")
        #print(self)
        #print()
        
        #print("A and B below prior to scaling by h or h**2")
        # Add boundaries
        # Room 1 has top and bottom constant 15, left constant 40, right Neumann with Room2
        room1_scale = room1.get_scale()
        room1_dims = room1.get_dims()
        r1_size = room1_scale*room1_dims
        # top, left, bottom, right ordering, not that it matters
        room1_boundaries = [[None, 15, np.array([0, r1_size[1]]), np.array([r1_size[0], r1_size[1]])],
                            [None, 40, np.array([0, 0]), np.array([0, r1_size[1]])],
                            [None, 15, np.array([0, 0]), np.array([r1_size[0], 0])],
                            [room2, 'N', np.array([r1_size[0], 0]), np.array([r1_size[0], r1_size[1]])]]
        room1.add_boundaries(room1_boundaries)
        room1.create_A()
        #print(f"Sparse and normal matrices are equal: {np.all(A_norm == A_sparse.toarray())}")
        self.send_sparse_matrix(room1.A, dest=1, shape_tag=0, data_tag=1)
        
        #print(room1.A)
        # Room 2 has top constant 40, bottom constant 5, left dirichlet with room1 on bottom half and constant 15
        # on top half of left, right constant 15 on bottom half and dirichlet with room3 on top half of right
        room2_scale = room2.get_scale()
        room2_dims = room2.get_dims() # array s.t. scale * dims gives coords of top right in (x, y)
        r2_size = room2_scale*room2_dims
        # top, left, bottom, right ordering, not that it matters
        room2_boundaries = [[None, 40, np.array([0, r2_size[1]]), np.array([r2_size[0], r2_size[1]])],
                            [room1, 'D', np.array([0, 0]), np.array([0, r2_size[1]//2])], # left bottom half
                            [None, 15, np.array([0, r2_size[1]//2]), np.array([0, r2_size[1]])], # left top half
                            [None, 5, np.array([0, 0]), np.array([r2_size[0], 0])],
                            [None, 15, np.array([r2_size[0], 0]), np.array([r2_size[0], r2_size[1]//4])], # right bottom quarter
                            [room4, 'D', np.array([r2_size[0], r2_size[1]//4]), np.array([r2_size[0], r2_size[1]//2])], # right with room 4
                            [room3, 'D', np.array([r2_size[0], r2_size[1]//2]), np.array([r2_size[0], r2_size[1]])]] # right top half
        room2.add_boundaries(room2_boundaries)
        room2.create_A()
        self.send_sparse_matrix(room2.A, dest=2, shape_tag=2, data_tag=3)
        #print(room1.A)
        #room2.V[0:2] = 100 # Can set values in room2 and see the boundary condition updated in room1!
        # Room 3 has top and bottom constant 15, left Neumann with Room 2, right constant 40
        room3_scale = room3.get_scale()
        room3_dims = room3.get_dims()
        r3_size = room3_scale*room3_dims
        # top, left, bottom, right ordering, not that it matters
        room3_boundaries = [[None, 15, np.array([0, r3_size[1]]), np.array([r3_size[0], r3_size[1]])],
                            [room2, 'N', np.array([0, 0]), np.array([0, r3_size[1]])],
                            [None, 15, np.array([0, 0]), np.array([r3_size[0], 0])],
                            [None, 40, np.array([r3_size[0], 0]), np.array([r3_size[0], r3_size[1]])]]
        room3.add_boundaries(room3_boundaries)
        room3.create_A()
        self.send_sparse_matrix(room3.A, dest=3, shape_tag=4, data_tag=5)
        # Room 4 has top constant 15, bottom constant 40, left Neumann with room2, right constant 15
        room4_scale = room4.get_scale()
        room4_dims = room4.get_dims()
        r4_size = room4_scale*room4_dims
        # top, left, bottom, right ordering, not that it matters
        room4_boundaries = [[None, 15, np.array([0, r4_size[1]]), np.array([r4_size[0], r4_size[1]])],
                            [room2, 'N', np.array([0, 0]), np.array([0, r4_size[1]])],
                            [None, 40, np.array([0, 0]), np.array([r4_size[0], 0])],
                            [None, 15, np.array([r4_size[0], 0]), np.array([r4_size[0], r4_size[1]])]]
        room4.add_boundaries(room4_boundaries)
        room4.create_A()
        self.send_sparse_matrix(room4.A, dest=4, shape_tag=6, data_tag=7)
        
    def add_room_to_plan(self, room, loc):
        # Add a room to the floor plan at the given location
        # Location given in global coordinates, not array coordinates
        room_dims = room.get_dims()
        # Location in array coordinates
        array_loc = loc * room.get_scale()
        # Get the top right coordinate of the new room, scaled to array coordinates
        top_right = (room_dims + loc) * room.get_scale()
        new_floor_shape = np.array(self.floor_plan.shape)
        # If this is out of bounds, increase the size of the floor plan
        if top_right[0] > self.floor_plan.shape[0]:
            new_floor_shape[0] = top_right[0]
        if top_right[1] > self.floor_plan.shape[1]:
            new_floor_shape[1] = top_right[1]
            
        new_plan = self.floor_plan.copy()
        # If the new floor plan is bigger, update the array
        if not np.all(new_floor_shape == self.floor_plan.shape):
            new_plan = np.zeros(new_floor_shape) - 1
            # Copy over old plan
            new_plan[0:self.floor_plan.shape[0], 0:self.floor_plan.shape[1]] = self.floor_plan
        # Add room values to new plan
        room_slice_x = slice(array_loc[0], top_right[0])
        room_slice_y = slice(array_loc[1], top_right[1])
        # If any values in the slice are not -1 (uninitialized), throw error
        #print(f"New room from {array_loc} to {top_right[0]}, {top_right[1]}")
        if np.any(new_plan[room_slice_x, room_slice_y] != -1):
            raise ValueError(f"Invalid location given, a room is already present at loc {loc}, array loc {array_loc}")
        new_plan[room_slice_x, room_slice_y] = room.get_temp_array()
        self.floor_plan = new_plan
        return new_plan
    
    def update_floor_plan(self):
        # Update the current floor plan with the temp values in the rooms
        for i in range(len(self.rooms)):
            room = self.rooms[i]
            room_array_loc = self.room_locs[i] * room.get_scale()
            top_right = (room.get_dims() + self.room_locs[i]) * room.get_scale()
            room_slice_x = slice(room_array_loc[0], top_right[0])
            room_slice_y = slice(room_array_loc[1], top_right[1])
            #print(f"Room {i+1}, floor plan portion: {self.floor_plan[room_slice_x, room_slice_y].shape}")
            self.floor_plan[room_slice_x, room_slice_y] = room.get_temp_array()
    
    # solve method
    def solve(self, iterations, omega):
        # Start solving in each room
        # Follow procedure as in project desc:
        # Given u1, u2, u3
        #   Solve u2_k+1 on room2
        #   Solve u1 and u3 k+1 on room1 and room3
        #   Relaxation: uk+1 = w*uk+1 + (1-w)*uk
        #   repeat
        for it in range(iterations):
            # Solve room2 first
            b2 = self.rooms[1].create_B()
            #print(b2)
            self.comm.Send([b2, MPI.DOUBLE], dest=2, tag=200+(it+1))
            #room2.solve(omega)
            #print(f"from master: {room2.V.shape}")
            self.comm.Recv(self.rooms[1].V, source=2, tag=2000+(it+1))
            #print(f"Received v vector from room 2: {room2.V}")
            # Solve all rooms but room 2 in parallel
            for r in range(len(self.rooms)):
                if r==1:
                    continue
                b = self.rooms[r].create_B()
                #print(b)
                self.comm.Send([b, MPI.DOUBLE], dest=r+1, tag=(100*(r+1))+(it+1))
            for r in range(len(self.rooms)):
                if r==1:
                    continue
                #recieve from rooms
                self.comm.Recv(self.rooms[r].V, source=r+1, tag=(1000*(r+1))+(it+1))
            
            
            #print(f"After iteration {it+1} room1 temps:")
            #print(room1.get_temp_array())
    
    def plot_heatmap(self, grid_size):
        # Display the temperature values as a heatmap
        data = self.floor_plan
        rows, cols = data.shape
        fig, ax = plt.subplots()

        valid_data = data[data != -1]
        vmin = valid_data.min()
        vmax = valid_data.max()
        range_values = vmax - vmin

        cmap = plt.colormaps.get_cmap('viridis')

        for i in range(rows):
            for j in range(cols):
                value = data[i, j]
                if value == -1:
                    continue
                x = i*grid_size
                y = j*grid_size
                #print(x, y)

                color = cmap((value - vmin) / range_values)

                rect = patches.Rectangle((x,y), grid_size, grid_size,
                                         #edgecolor='black', 
                                         facecolor=color,
                                         alpha=0.8)
                ax.add_patch(rect)

        ax.set_xlim(-0.1, rows * grid_size + 0.1)
        ax.set_ylim(-0.1, cols * grid_size + 0.1)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Value')

        plt.tight_layout()
        plt.title(fr"Heat distribution with $\Delta x={delta_x}$ after {iterations} iterations")
        plt.show()
    
    # Repr output
    def __repr__(self):
        #max_len = max([len(str(val)) for val in self.floor_plan.flatten()])
        str_rep = ''
        for i in range(self.floor_plan.shape[1]-1, -1, -1):
            row_str = ''
            for val in self.floor_plan[:, i]:
                if val == -1:
                    row_str += "#".ljust(2, ' ')
                else:
                    row_str += "0".ljust(2, ' ')
            str_rep += row_str + '\n'
        return str_rep

if __name__ == '__main__':
    # create apartment if on rank 0 process
    # Create the rooms first, telling each one its size
    # Apartment holds array of rooms and x,y positions of all of them
    #room1 = Room(size m, n)
    #room2 = Room(size x, y)
    #room1.create_boundaries(array of boundaries for room 1)
    # Then arrange the rooms into an apartment
    
    commMain = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = commMain.Get_rank()
    delta_x = 1/10
    layout = 'project3'
    
    iterations = 25
    omega = 0.8
    if layout == 'project3':
        if rank== 0:
            apartment = Apartment(commMain)
            apartment.initialize_apartment_proj3(delta_x)
            apartment.solve(iterations, omega)
            # SOME PLOTTING STUFF
            apartment.update_floor_plan()
            # Plot the heat values
            apartment.plot_heatmap(delta_x)
            
        if rank == 1:
            room1 = Room(rank, np.array([1, 1]), delta_x, commMain)
            #let this work
            #commMain.Recv(room1.A, source=0, tag=0)
            room1.A = room1.recv_sparse_matrix(source=0, shape_tag=0, data_tag=1)
            #print(f"current room A {rank}: \n{room1.A.toarray()}")
            room1.solve(iterations, omega)
        if rank == 2:
            room2 = Room(rank, np.array([1, 2]), delta_x, commMain)
            #commMain.Recv(room2.A, source=0, tag=1)
            room2.A = room2.recv_sparse_matrix(source=0, shape_tag=2, data_tag=3)
            #print(f"current room A {rank}: \n{room2.A.toarray()}")
            room2.solve(iterations, omega)
        if rank == 3:
            room3 = Room(rank, np.array([1, 1]), delta_x, commMain)
            #commMain.Recv(room3.A, source=0, tag=2)
            room3.A = room3.recv_sparse_matrix(source=0, shape_tag=4, data_tag=5)
            #print(f"current room A {rank}: \n{room3.A.toarray()}")
            room3.solve(iterations, omega)
        else:
            pass
        #print(apartment)
    elif layout == 'project3a':
        #omega = 0.8
        if rank== 0:
            apartment = Apartment(commMain)
            apartment.initialize_apartment_proj3a(delta_x)
            apartment.solve(iterations, omega)
            # SOME PLOTTING STUFF
            apartment.update_floor_plan()
            # Plot the heat values
            apartment.plot_heatmap(delta_x)
        elif rank == 1:
            room1 = Room(rank, np.array([1, 1])*2, delta_x, commMain)
            #let this work
            #commMain.Recv(room1.A, source=0, tag=0)
            room1.A = room1.recv_sparse_matrix(source=0, shape_tag=0, data_tag=1)
            #print(f"current room A {rank}: \n{room1.A.toarray()}")
            room1.solve(iterations, omega)
        elif rank == 2:
            room2 = Room(rank, np.array([1, 2])*2, delta_x, commMain)
            #commMain.Recv(room2.A, source=0, tag=1)
            room2.A = room2.recv_sparse_matrix(source=0, shape_tag=2, data_tag=3)
            #print(f"current room A {rank}: \n{room2.A.toarray()}")
            room2.solve(iterations, omega)
        elif rank == 3:
            room3 = Room(rank, np.array([1, 1])*2, delta_x, commMain)
            #commMain.Recv(room3.A, source=0, tag=2)
            room3.A = room3.recv_sparse_matrix(source=0, shape_tag=4, data_tag=5)
            #print(f"current room A {rank}: \n{room3.A.toarray()}")
            room3.solve(iterations, omega)
        elif rank == 4:
            room4 = Room(rank, np.array([1, 1]), delta_x, commMain)
            #commMain.Recv(room3.A, source=0, tag=2)
            room4.A = room4.recv_sparse_matrix(source=0, shape_tag=6, data_tag=7)
            #print(f"current room A {rank}: \n{room4.A.toarray()}")
            room4.solve(iterations, omega)
