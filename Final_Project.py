# ENGG*3130 - Modeling Complex Systems
## Final Project

from __future__ import print_function
import io
import networkx as nx
import numpy as np
import json
import string
import matplotlib.pyplot as plt
from itertools import combinations 

# Import MapQuest API Libraries from https://github.com/MapQuest/directions-api-python-client somehow
from RouteOptions import RouteOptions
from RouteService import RouteService

# Import Excel libraries
from openpyxl import Workbook
from openpyxl import load_workbook

# Import Google VRP libraries
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

#%matplotlib inline

data_file = "./data/1_02.xlsm" #Enter the data file for the day to schedule
driver = '1642'
sleeman_location = "551 Clair Rd W, Guelph, ON N1L 1E9"
    
wb = load_workbook(data_file, read_only=False, keep_vba=True)
ws = wb.active
deliveries = {}
load_weights = {}

options = RouteOptions()
service = RouteService('UaKqbaGJv9wQGxdOe95QTTSWa1ldsj9e')

# CHANGE TO NOT SPLIT UP INTO DRIVERS, ONLY KEEPING THIS NOW FOR EASE OF DEBUGGING
for row in range(1, len(ws['A'])+1):
    if ('delivery' in str(ws['J'+str(row)].value)):
        if str(ws['A'+str(row)].value) in deliveries.keys():
            deliveries[str(ws['A'+str(row)].value)].append(str(ws['R'+str(row)].value).replace('#',''))
            load_weights[float(ws['A'+str(row)].value)].append(float(ws['AK'+str(row)].value))
        else:
            deliveries[str(ws['A'+str(row)].value)] = [str(ws['R'+str(row)].value).replace('#','')]
            load_weights[float(ws['A'+str(row)].value)] = [float(ws['AK'+str(row)].value)]

def get_weighted_dist_matrix(locations, loads, factor_weight):

    route_matrix = service.routeMatrix(locations=locations, oneToMany=True)
    dist_matrix = route_matrix['distance']
    weighted_dist_matrix = dist_matrix

    i = 0;
    for load_weight in loads:
        weighted_dist_matrix[i] = dist_matrix[i]*(1/load_weight)*factor_weight
        i+=1
    return weighted_dist_matrix


def get_weighted_time_matrix(locations, loads, factor_weight):

    route_matrix = service.routeMatrix(locations=locations, oneToMany=True)

    time_matrix = route_matrix['time']
    weighted_time_matrix = time_matrix

    i = 0;
    for load_weight in loads:
        weighted_time_matrix[i] = time_matrix[i] * (1/load_weight) * factor_weight
        i+=1
    return weighted_time_matrix


def calc_edges(locations, loads=None, factor=None, factor_weight=None):
    """Calculate the edge weight"""

    if factor is None:
        routeMatrixRaw = service.routeMatrix(locations=locations, oneToMany=True)
        routeMatrix = routeMatrixRaw['distance']
  
    elif factor is 'dist':
        routeMatrix = get_weighted_dist_matrix(locations=locations, loads=loads, factor_weight=factor_weight)
    
    elif factor is 'time':
        routeMatrix = get_weighted_time_matrix(locations=locations, loads=loads, factor_weight=factor_weight)
    
    return routeMatrix
  
def create_driver_graph():
    G = nx.DiGraph()
    drive_times_all = {}
    drive_times = {}
    
    ####### Single Driver Parsing #######
    #nodes = deliveries[driver] 

    ####### All Driver Parsing #######
    nodes = [y for x in deliveries.values() for y in x]

    nodes.insert(0, sleeman_location) # Add the origin destination

    for n in range(0, len(nodes)): 
        cycled_nodes = (nodes[-n:] + nodes[:-n])
        routeMatrix = calc_edges(cycled_nodes)

        for i in range(1, len(cycled_nodes)):
            #print cycled_nodes[0]+' ~ AND ~ '+cycled_nodes[i]
            #import ipdb
            #ipdb.set_trace()
            #print str(cycled_nodes[0]) +" --------- "+ str(cycled_nodes[i]) +" --------- "+ str(routeMatrix[i])
            drive_times_all[(cycled_nodes[0], cycled_nodes[i])] = float(routeMatrix[i])

    all_combinations = list(combinations(nodes, 2))
    for n in range(len(all_combinations)):
        val1 = drive_times_all[all_combinations[n]]
        reversed_combination = all_combinations[n][::-1]
        val2 = drive_times_all[reversed_combination]
        avg = (val1+val2)/2.0
        drive_times[all_combinations[n]] = round(avg, 1)
        drive_times[reversed_combination] = round(avg, 1)

    G.add_nodes_from(nodes)
    G.add_edges_from(drive_times)

    return G, drive_times, nodes

def display_graph(G, edges):
    nx.draw(G,
        node_color='b',
        node_size=2000,
        with_labels=True,
        font_size=10)
    nx.draw_networkx_edge_labels(G, edge_labels=edges, pos=nx.spring_layout(G), font_size=8)
    plt.show()

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    distance_matrix = []
    distance_matrix_list = []

    graphs_and_edges = create_driver_graph()
    drive_times = graphs_and_edges[1]
    delivery_nodes = graphs_and_edges[2]

    for n in range(len(delivery_nodes)-1):
        del distance_matrix_list[:]
        for i in range(len(delivery_nodes)-1):
            search_tuple = (delivery_nodes[n], delivery_nodes[i])
            if search_tuple in drive_times:
                distance_matrix_list.append(int(drive_times[search_tuple]))
            else:
                distance_matrix_list.append(int(0))
        #print distance_matrix_list # Debugging line
        distance_matrix.append(list(distance_matrix_list))

    data['distance_matrix'] = distance_matrix
    data['num_vehicles'] = 4 # Not static, should be parsed from command line
    data['depot'] = 0 # Should be static?

    return data

"""https://developers.google.com/optimization/routing/vrp"""
def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))

"""https://developers.google.com/optimization/routing/vrp"""
def main():
    """Separates out the location list of a single truck"""
    #Enter number for truck to get locations for
    one_truck_locations = deliveries[driver]
    one_truck_load_weights = load_weights[float(driver)]

    #graph_and_edges = create_driver_graph()
    #routeGraph = graph_and_edges[0]
    #edges = graph_and_edges[1]
    #display_graph(routeGraph, edges)                                                   

    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()
    #print data

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC) # FIRST SOLUTION STRATEGY WILL BE USED TO COMPARE ALGORITHMS

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)

if __name__ == '__main__':
    main()
