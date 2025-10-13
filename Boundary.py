import numpy as np

def dirilchlet_condition():
    pass

class Boundary():
    def __init__(self, side, tp, location, neighbor, value):
        """
        string  side:       Side of the square located ("left", "right", "top" or "bottom")
        string  tp:         Mathematical type of the boundary ("Dirichlet" or "Neumann")
        list    location:   2 points that represents the limits of the (left to right or top to bottom)
        room    neighbor:   Room to which the boundary is connected
        float   value   :   Boundary value
        """
        self.side = side
        self.math_type = tp
        self.location = location
        self.neighbor = neighbor
        self.value = value

    def boundary_indices(self, grid_size, room_size):
        """
        Get grid indices that are contained in this boundary segment
        """
        total_x = int(room_size[0]/grid_size) + 1
        total_y = int(room_size[1]/grid_size) + 1

        (x0, y0), (x1, y1) = self.location

        x_start, x_end = round(x0 / grid_size), round(y0 / grid_size)
        y_start, y_end = round(x1 / grid_size), round(y1 / grid_size)

        #x_start, x_end = sorted((i0, i1))
        #y_start, y_end = sorted((j0, j1))

        boundaries = {
        "left":   [(0.0, round(y*grid_size, 3)) for y in range(y_start, y_end + 1)],
        "right":  [(room_size[0], round(y*grid_size, 3)) for y in range(y_start, y_end + 1)],
        "bottom": [(round(x*grid_size, 3), 0.0) for x in range(x_start, x_end + 1)],
        "top":    [(round(x*grid_size, 3), room_size[1]) for x in range(x_start, x_end + 1)]
        }
        
        return boundaries[self.side]


boundar1 = Boundary("left", "D", ((0, 2),(0, 1)), None, 5)
print(boundar1.boundary_indices(0.05, (2, 2)))

        
        
    