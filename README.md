# bmed8813-path-planning

## Algorithm to plan the shortest route for Streach Robot to find an object in different locations. (Robot Finder Project)

1. The tsp_solver.py has a class named Path_finder. It can be used to create objects that solve the Traveling Salesman Problem (TSP) to find the shortest path to search for an objects accounting for the likelihood and distance of the waypoints. 
2. The final_class_test Jupyter Notebook has some examples on how to import the script and being able to use the class.
3. The TSP solver used synthetic data storaged in csv files. Please make sure the labels of this data are aligned with the waypoints order in the numpy arrays. You need to pass an Nx2 numpy array with the coordinates of the waypoints. The first row MUST be the inital position of the robot.
4. The output of the main function of the class (solve_tsp()) returns a list with the indexes of the rows of the waypoints' numpy array excluding the initial position of the robot
