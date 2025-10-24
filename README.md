# NUMN21-Project3
Repository for Project 3 in Advanced Course in Numerical Algorithms with Python/SciPy, at Lund University.

Group 6: Martin Hindle, Nicolas Vega, Colin Harmon \
Work done: \
All three group members made meaningful contributions to the project. We began by defining the class structure and then worked on much of the code in person.

Invidivual contributions:\
Martin: Structure, Parallelism, Math checking, Bug fixing \
Nicolas: Structure, Boundary class, Heatmap plotting, Bug fixing \
Colin: Structure, Matrix/boundary setup from boundary conditions, Equation solving, Parallelism, Command line inputs, Bug fixing, Fix for heat flow issues

## Running the project
The project can be run by using mpi to run the Apartment.py file (command depending on OS). 

Arguments are specified using argparse, so to see the help string add the argument '-h'. These can include the layout, delta x to use for gridsize, number of iterations, relaxation parameter omega, the heater temperatures (either a single value or a list of values, one for each room), whether to average along neutral borders, whether to plot the output or not, whether to show a secondary plot with highlights to reveal heat flow better, and whether to show average temperatures by room at the end.

There are 7 layouts possible:
  - project3: The default layout for project 3
  - project3a: The extension for project 3, with one open border between room 2 and room 4.
  - project3a_connected: The extension with room 4 connected to both rooms 2 and 3.
  - single_room: A single room demo.
  - double_room: Two rooms connected across one border, solved in parallel.
  - sequential: The same two rooms, solved sequentially without MPI.
  - sequential_single: The same two room setup, solved as one room without a border to check correct solution.

Note that due to the way rooms are solved, a minimum of 4 processes will be needed (-n 4) for project 3 and 5 for project 3 extension. This is enforced by the code.

Also, generating the temperatures is quite fast, even for large delta x values, but plotting is not. To assess runtime without plotting, use the parameter -p 0 to disable the heat plot.

## Project 3
### Task 1
See matrices pdf for the A matrices generated for task1

### Task 2
Program run with 10 iterations, 1/20 delta x, and 0.8 omega
The heating in the flat is poorly distributed. Along the edges with heaters or the windows, it is very unpleasant at 35-40 degrees along heaters and less than 15 near the window. There are tiny places in rooms 1 and 3 where the temperature is nice, but for the majority of the areas it is too hot or too cold. We would say the heating is not adequate for these reasons.

### Task 3
To plot the temperature distribution, simply run the program with -p 1 or leaving it to the default of 1. To add a second plot with highlights around the mid temperatures, use --highlights 1 or leave it to the default of 1.

### Task 4
To vary the parameters, use the arguments provided.

## Project 3 Extension
Here, we notice that the temperatures seem more comfortable across more of the rooms, with large areas in the temperature range 15-30. With room 4 fully connected to the others, there is a reasonably large area between rooms 2, 3 and 4 that is roughly 25 degrees, which is a notable improvement over the initial apartment. Of course, because the heaters and windows are so extreme in temperature, the edges are still uncomfortable.
