# Class for the apartment, which knows about it's rooms
import numpy as np
import scipy
from Room import Room
from mpi4py import MPI

class Apartment():
    # Know about its rooms, how many, how they are connected
    # Needs to be told where the rooms are and connect them
    # NOTE: In the floor plan array, the apartment is represented so that you
    # can build the rooms using (x, y) values as you might graph on a paper. This means
    # it actually is kind of horizontal in the array, but should work.
    def __init__(self):
        self.rooms = []
        self.room_locs = []
        # Floor plan contains an array of values representing the apartment
        # Rooms are then inserted at specified locations in the floor plan
        # and take up the specified size and shape in the array
        # NOTE: (x, y) are measured from bottom left of a room
        # so if I add a room at (2, 1) it adds the bottom left of the room at (2, 1)
        self.floor_plan = np.array([[-1]])
    
    # Function to create an apartment that matches the design in project 3
    def initialize_apartment_proj3(self, delta_x):
        # Create the apartment in project 3, containing one room 1x1 connected at the lower left
        # of a larger room 2x1, with another 1x1 room connected on the top right.
        # Add room of size 1x1 at origin
        rank = 0
        room1 = Room(rank, np.array([1, 1]), delta_x)
        rank += 1
        room2 = Room(rank, np.array([1, 2]), delta_x)
        rank += 1
        room3 = Room(rank, np.array([1, 1]), delta_x)
        print("Initial plan:")
        print(self)
        print()
        # Add room 1 at (0, 0)
        room1_loc = np.array([0, 0])
        self.room_locs.append(room1_loc)
        self.add_room_to_plan(room1, room1_loc)
        self.rooms.append(room1)
        print("Room 1 added:")
        print(self)
        print()
        # Add room 2 at (1, 0)
        room2_loc = np.array([1, 0])
        self.room_locs.append(room2_loc)
        self.add_room_to_plan(room2, room2_loc)
        self.rooms.append(room2)
        print("Room 2 added:")
        print(self)
        print()
        # Add room 3 at (2, 1)
        room2_loc = np.array([2, 1])
        self.room_locs.append(room2_loc)
        self.add_room_to_plan(room3, room2_loc)
        self.rooms.append(room3)
        print("Room 3 added:")
        print(self)
        print()
        print("A and B below prior to scaling by h or h**2")
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
        print(room1.A)
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
        #print(room3.A)
        # After boundaries are defined, create boundary vectors B
        room1.create_B()
        # Try modifying data in room3
        #room3.V[0:2] = 100 # We see the last two elements in room2 B vector changing!
        room2.create_B()
        room3.create_B()
        print(room2.B)
        # Try some math
        vtest = scipy.linalg.solve(room2.A, room2.B)
        print(f"Solution to Ax=B: {vtest}")
        # Try an iteration of solving all rooms
        print(f"Before solve room1 temps:\n{room1.get_temp_array()}")
        self.solve(3, omega=0.8)
        
        # Update floor plan and print
        self.update_floor_plan()
        print(self.floor_plan)
        
        
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
        room_slice_x = slice(array_loc[0], array_loc[0]+top_right[0])
        room_slice_y = slice(array_loc[1], array_loc[1]+top_right[1])
        # If any values in the slice are not -1 (uninitialized), throw error
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
            print(f"Room {i+1}, floor plan portion: {self.floor_plan[room_slice_x, room_slice_y].shape}")
            self.floor_plan[room_slice_x, room_slice_y] = room.get_temp_array()
    
    def create_apartment(self):
        # Take some measure of number of rooms, their sizes, where they are
        # Create 
        pass
    
    # solve method
    def solve(self, iterations, omega):
        # Start solving in each room
        # Follow procedure as in project desc:
        # Given u1, u2, u3
        #   Solve u2_k+1 on room2
        #   Solve u1 and u3 k+1 on room1 and room3
        #   Relaxation: uk+1 = w*uk+1 + (1-w)*uk
        #   repeat
        room1, room2, room3 = self.rooms
        for it in range(iterations):
            # Solve room2 first
            room2.solve(omega)
            # Solve rooms 1 and 3 next (in parallel eventually)
            room1.solve(omega)
            room3.solve(omega)
            print(f"After iteration {it+1} room1 temps:")
            print(room1.get_temp_array())
    
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
    delta_x = 1/2
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    if rank == 0:
        print(f"Rank {rank} creating apartment")
        apartment = Apartment()
        apartment.initialize_apartment_proj3(delta_x)
        #print(apartment)
    else:
        print(f"Hello from rank {rank}")