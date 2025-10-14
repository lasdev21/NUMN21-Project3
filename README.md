# NUMN21-Project3
Repository for Project 3 in Advanced Course in Numerical Algorithms with Python/SciPy, at Lund University.

I started the code that can add rooms to an apartment. Currently you build rooms and pass them to the add_room_to_plan function. \
Run Apartment.py to see current output. Check out initialize_apartment_project3 to see how it is currently building an apartment.

Apartment on master thread rank == 0, then room i on rank i.
Room 2 checks if iteration is 0 then begin solve then send to ranks 1,3.

Rooms 1,3 wait for room 2 information then solve then send back to room 2.
