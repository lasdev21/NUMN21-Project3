# NUMN21-Project3
Repository for Project 3 in Advanced Course in Numerical Algorithms with Python/SciPy, at Lund University.

Group 6: Martin Hindle, Nicolas Vega, Colin Harmon \
Work done: \
All three group members made meaningful contributions to the project. We began by defining the class structure and then worked on much of the code in person.

Invidivual contributions:\
Martin: Structure, Parallelism, Math checking, Bug fixing \
Nicolas: Structure, Boundary class, Heatmap plotting, Bug fixing \
Colin: Structure, Matrix/boundary setup from boundary conditions, Equation solving, Parallelism, Bug fixing

## Running the project
The project can be run by using mpi to run the Apartment.py file (command depending on OS). 

Arguments are specified using argparse, so to see the help string add the argument '-h'. These can include the layout, delta x to use for gridsize, number of iterations, relaxation parameter omega, and whether to plot the output or not.

Note that due to the way rooms are solved, a minimum of 4 processes will be needed (-n 4) for project 3 and 5 for project 3 extension.

## Project 3
### Task 1
See matrices pdf for the A matrices generated for task1

### Task 2
Program run with 10 iterations, 1/20 delta x, and 0.8 omega
The heating in the flat is not so bad in the middle of the rooms. It seems the temperature is between 15 and 30 degrees, which is livable. However, it is poorly distributed and along the edges with heaters or the windows, it is very unpleasant at 35-40 degrees along heaters and less than 15 near the window. We would say the heating is not adequate for these reasons.

### Task 3
To plot the temperature distribution, simply run the program with -p 1 or leaving it to the default of 1.

### Task 4
To vary the parameters, use the arguments provided.

## Project 3 Extension
Here, we notice that the temperatures seem more comfortable across more of the rooms, with temperatures from 20-30 in most room centers. Of course, because the heaters and windows are so extreme in temperature, the edges are still uncomfortable.

Note that we seem to get an error when delta x is specified at 100 for the extension. Try with project 3 instead. The code will generate the solution quite quickly, but plotting is slow. To generate without plotting to assess runtime, use -p 0 as a parameter.
