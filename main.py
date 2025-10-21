import numpy as np
from Room import Room
from Apartment import Apartment
from mpi4py import MPI
import argparse
import os, sys

if __name__ == '__main__':
    # create apartment if on rank 0 process
    # The apartment creates the room layout and
    # holds array of rooms and temperatures of all of them
    
    commMain = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = commMain.Get_rank()
    
    # Temporarily remove printing from all but master process to stop multiple help/error strings from argparse
    if rank != 0:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    # Argparse
    # Accepts arguments for layout, delta_x, iterations, omega
    parser = argparse.ArgumentParser(description="Modelling Laplace heat equation on room structure using parallel MPI processes")
    parser.add_argument("layout", choices=['project3', 'project3a', 'project3a_connected', 'single_room', 'double_room', 'sequential'], 
                        help="""Specify the room layout you want to solve on.""")
    parser.add_argument("-dx", "--delta_x", type=int, default=10, 
                        help="The reciprocal of the gridwidth to choose. I.e. a value of 10 means 1/10 delta x.")
    parser.add_argument("-it", "--iterations", type=int, default=10,
                        help="The number of iterations to perform.")
    parser.add_argument("-w", "--omega", type=float, default=0.8,
                        help="The relaxation parameter omega.")
    parser.add_argument("-heat", "--heater_temp", type=float, nargs='+', default=[40.],
                        help="""The temperature of heaters across the rooms. If a single value is given, it is used \
                            for all heaters. If multiple values are given, one must be given for each room and is used for heaters \
                            in that room. Default 40.""")
    parser.add_argument("-a", "--avg_borders", type=int, choices=[0, 1], default=0,
                        help="Boolean value for replacing constant 15 temperature neutral borders with a value proportional to the avg temp along the border. Defaults to 0 (False).")
    parser.add_argument("-p", "--plot", type=int, choices=[0, 1], default=1,
                        help="Boolean value for determining plotting of outputs. Defaults to 1 (True). Note that for large (~100) delta x, plotting is very slow.")
    parser.add_argument("--print_avgs", type=int, choices=[0, 1], default=1,
                        help="Boolean value for displaying average temperatures of rooms once complete. Defaults to 1 (True).")
    args = parser.parse_args()
    
    delta_x = 1/args.delta_x
    layout = args.layout
    iterations = args.iterations
    omega = args.omega
    heater_temp = args.heater_temp
    avg_borders = args.avg_borders
    if avg_borders:
        print("Using averaging along neutral borders")
    make_plot = args.plot
    print_avg = args.print_avgs
    
    # Restore stdout and stderr
    if rank != 0:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
    #print(f"On process {rank}")
    #print(f"Got args:\nlayout:{layout}\ndelta:{delta_x}\niters:{iterations}\nomega:{omega}")
    # Begin program operation
    # Create apartment on master process
    if rank == 0:
        apartment = Apartment(commMain)
    # Project3 base
    if layout == 'project3':
        # Make sure we have enough processes
        assert commMain.Get_size() >= 4, "Too few processes, please run again with at least 4"
        if rank== 0:
            #apartment = Apartment(commMain)
            apartment.initialize_apartment_proj3(delta_x, heater_temp, avg_borders)
            apartment.solve(iterations, omega)
        elif rank == 1:
            room1 = Room(rank, np.array([1, 1]), delta_x, commMain)
            #let this work
            #commMain.Recv(room1.A, source=0, tag=0)
            room1.A = room1.recv_sparse_matrix(source=0, shape_tag=0, data_tag=1)
            #print(f"current room A {rank}: \n{room1.A.toarray()}")
            room1.solve(iterations, omega)
        elif rank == 2:
            room2 = Room(rank, np.array([1, 2]), delta_x, commMain)
            #commMain.Recv(room2.A, source=0, tag=1)
            room2.A = room2.recv_sparse_matrix(source=0, shape_tag=2, data_tag=3)
            #print(f"current room A {rank}: \n{room2.A.toarray()}")
            room2.solve(iterations, omega)
        elif rank == 3:
            room3 = Room(rank, np.array([1, 1]), delta_x, commMain)
            #commMain.Recv(room3.A, source=0, tag=2)
            room3.A = room3.recv_sparse_matrix(source=0, shape_tag=4, data_tag=5)
            #print(f"current room A {rank}: \n{room3.A.toarray()}")
            room3.solve(iterations, omega)
    # Extension
    elif layout == 'project3a' or layout == 'project3a_connected':
        # Make sure we have enough processes
        assert commMain.Get_size() >= 5, "Too few processes, please run again with at least 5"
        scalar = 2
        if rank== 0:
            #apartment = Apartment(commMain)
            if layout == 'project3a':
                apartment.initialize_apartment_proj3a(delta_x, scalar, heater_temp, avg_borders)
            else:
                apartment.initialize_apartment_proj3a_connected(delta_x, scalar, heater_temp, avg_borders)
            apartment.solve(iterations, omega)
            # Update the floor plan with final temps
            #apartment.update_floor_plan()
            # Plot the heat values
            #if make_plot:
                #apartment.plot_heatmap(delta_x)
            #else:
                #print("Completed process, plotting disabled")
        elif rank == 1:
            room1 = Room(rank, np.array([1, 1])*scalar, delta_x, commMain)
            #let this work
            #commMain.Recv(room1.A, source=0, tag=0)
            room1.A = room1.recv_sparse_matrix(source=0, shape_tag=0, data_tag=1)
            #print(f"current room A {rank}: \n{room1.A.toarray()}")
            room1.solve(iterations, omega)
        elif rank == 2:
            room2 = Room(rank, np.array([1, 2])*scalar, delta_x, commMain)
            #commMain.Recv(room2.A, source=0, tag=1)
            room2.A = room2.recv_sparse_matrix(source=0, shape_tag=2, data_tag=3)
            #print(f"current room A {rank}: \n{room2.A.toarray()}")
            room2.solve(iterations, omega)
        elif rank == 3:
            room3 = Room(rank, np.array([1, 1])*scalar, delta_x, commMain)
            #commMain.Recv(room3.A, source=0, tag=2)
            room3.A = room3.recv_sparse_matrix(source=0, shape_tag=4, data_tag=5)
            #print(f"current room A {rank}: \n{room3.A.toarray()}")
            room3.solve(iterations, omega)
        elif rank == 4:
            room4 = Room(rank, np.array([0.5, 0.5])*scalar, delta_x, commMain)
            #commMain.Recv(room3.A, source=0, tag=2)
            room4.A = room4.recv_sparse_matrix(source=0, shape_tag=6, data_tag=7)
            #print(f"current room A {rank}: \n{room4.A.toarray()}")
            room4.solve(iterations, omega)
    elif layout == 'single_room':
        # Make sure we have enough processes
        assert commMain.Get_size() >= 2, "Too few processes, please run again with at least 2"
        scalar = 1
        if rank== 0:
            #apartment = Apartment(commMain)
            apartment.simple_room_test(delta_x, scalar, heater_temp)
            apartment.solve_test(iterations, omega)
            # Update the floor plan with final temps
            #apartment.update_floor_plan()
            # Plot the heat values
            #if make_plot:
                #apartment.plot_heatmap(delta_x)
            #else:
                #print("Completed process, plotting disabled")
        elif rank == 1:
            room1 = Room(rank, np.array([1, 1])*scalar, delta_x, commMain)
            room1.A = room1.recv_sparse_matrix(source=0, shape_tag=0, data_tag=1)
            #print(f"current room A {rank}: \n{room1.A.toarray()}")
            room1.solve(iterations, omega)
    elif layout == 'double_room':
        # Make sure we have enough processes
        assert commMain.Get_size() >= 3, "Too few processes, please run again with at least 2"
        scalar = 1
        if rank== 0:
            apartment.double_room_test(delta_x, scalar, heater_temp)
            apartment.double_solve_test(iterations, omega)
            # Update the floor plan with final temps
            #apartment.update_floor_plan()
            # Plot the heat values
            #if make_plot:
                #apartment.plot_heatmap(delta_x)
            #else:
                #print("Completed process, plotting disabled")
        elif rank == 1:
            room1 = Room(rank, np.array([1, 1])*scalar, delta_x, commMain)
            room1.A = room1.recv_sparse_matrix(source=0, shape_tag=0, data_tag=1)
            #print(f"current room A {rank}: \n{room1.A.toarray()}")
            room1.solve(iterations, omega)
        elif rank == 2:
            room2 = Room(rank, np.array([1, 1])*scalar, delta_x, commMain)
            room2.A = room2.recv_sparse_matrix(source=0, shape_tag=2, data_tag=3)
            #print(f"current room A {rank}: \n{room1.A.toarray()}")
            room2.solve(iterations, omega)
    elif layout == 'sequential':
        # Make sure we have enough processes
        assert commMain.Get_size() >= 1, "Too few processes, please run again with at least 1"
        scalar = 1
        if rank== 0:
            #apartment = Apartment(commMain)
            apartment.sequential_room_test(delta_x, scalar, heater_temp)
            apartment.sequential_solve(iterations, omega)
            
    # Regardless of which layout we use, plotting is decided the same way
    # Done from master process
    if rank == 0:
        # Update the floor plan with final temps
        apartment.update_floor_plan()
        # Display the average temperatures
        if print_avg:
            apartment.print_average_temps()
        # Plot the heat values
        if make_plot:
            apartment.plot_heatmap(delta_x, iterations)
        else:
            print("Completed process, plotting disabled")