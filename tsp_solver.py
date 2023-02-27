import numpy as np
from itertools import permutations
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Define a function to calculate the length of a tour
def tour_length(tour, graph):
    length = 0
    for i in range(len(tour) - 1):
        #print(graph[tour[i]][tour[i+1]])
        length += graph[tour[i]][tour[i+1]]
    return length

#Simple TSP solver
def simple_tsp_tour_solver(graph, visit_nodes):
    
    # Generate all possible permutations of visit_nodes
    visit_permutations = list(permutations(visit_nodes))

    # Initialize the shortest tour and its length
    shortest_tour = None
    shortest_length = np.inf

    # Iterate over all permutations of visit_nodes
    for visit_perm in visit_permutations:
        
        # Insert the start node 'R' at the beginning of the tour
        tour = [0] + list(visit_perm)

        # Update the shortest tour and its length
        #print(tour, tour_length(tour, graph))
        if tour_length(tour, graph) < shortest_length:
            shortest_tour = tour
            shortest_length = tour_length(tour, graph)
    
    # Return the final tour and its length
    return (shortest_tour, shortest_length)

def graph_visualization(nodes):
    #print(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    pairs = list(combinations(nodes, 2))
    for pair in pairs:
        G.add_edge(pair[0], pair[1])
    return G

def solution_visualization(solution_nodes):
    G = nx.Graph()
    G.add_nodes_from(solution_nodes)
    for i in range(0, len(solution_nodes) -1):
        G.add_edge(solution_nodes[i], solution_nodes[i + 1])
    return G

def generate_path(nodes, index_ls):
    path = []
    for index in index_ls:
        path.append(nodes[index])
    return path

def generate_path_coor(coor_matrix, index_ls):
    path = []
    #print("matrix:", coor_matrix)
    for index in index_ls:
        #print(index)
        #print(coor_matrix[index,0])
        path.append((coor_matrix[index, 0], coor_matrix[index, 1]))
    return path

#Euclidean distance calculator
def eu_dist(coor_matrix):
    #print(coor_matrix[:, np.newaxis])
    sq_sub = (coor_matrix[:, np.newaxis] - coor_matrix)**2
    #print(sq_sub)
    #print('-------------------')
    eu_dist = np.sqrt(np.sum(sq_sub, axis=2))
    #print(eu_dist)
    return eu_dist

def cost_matrix_calc(coor_matrix, prob, w_l, w_d):
    #Calculate eculidean distance pairwise between nodes
    sq_sub = (coor_matrix[:, np.newaxis] - coor_matrix)**2
    eu_dist = np.sqrt(np.sum(sq_sub, axis=2))
    #Finds the largest distance to nomalize the matrix values
    max_val = np.amax(eu_dist)
    graph_norm = eu_dist / max_val
    #Multiply the likelihood matrix and the distance matrix by their weights
    graph_w = graph_norm * w_d
    prob_w = prob * w_l
    #Calculate the cost of from one node to another using the cost formula
    cost_graph = graph_w - prob_w.reshape(prob_w.shape[1], prob_w.shape[0])
    return cost_graph


file_name = 'synthetic_data.csv'
df = pd.read_csv(file_name)
ocurr_arr = df.to_numpy()
ocurr_arr = ocurr_arr.reshape(ocurr_arr.shape[1], ocurr_arr.shape[0])
#print(ocurr_arr)
total_val = ocurr_arr.size
unique_val = np.unique(ocurr_arr) 
print(unique_val)

likeli_arr = np.zeros(len(unique_val), 1)
for i in range(0, len(unique_val)):
    curr_arr = ocurr_arr[unique_val[i]]
    likelihood = curr_arr.size / total_val
    likeli_arr[i,0] = likelihood
