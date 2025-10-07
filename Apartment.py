# Class for the apartment, which knows about it's rooms
import Room

class Apartment():
    # Know about its rooms, how many, how they are connected
    # Needs to be told where the rooms are and connect them
    def __init__(self):
        # Maybe create rooms
        self.rooms = []
    
    def create_apartment(self):
        # Take some measure of number of rooms, their sizes, where they are
        # Create 
        pass
    
    # solve method
    def solve(self):
        # Start solving in each room
        pass


if __name__ == '__main__':
    # create apartment if on rank 0 process
    # Create the rooms first, telling each one its size
    # Apartment holds array of rooms and x,y positions of all of them
    room1 = Room(size m, n)
    room2 = Room(size x, y)
    room1.create_boundaries(array of boundaries for room 1)
    # Then arrange the rooms into an apartment
    apartment = Apartment()