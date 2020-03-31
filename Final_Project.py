# ENGG*3130 - Modeling Complex Systems
## Final Project

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

#%matplotlib inline

data_file = "./data/1_02.xlsm" #Enter the data file for the day to schedule
driver = '1642'
sleeman_location = "551 Clair Rd W, Guelph, ON N1L 1E9"
    
wb = load_workbook(data_file, read_only=False, keep_vba=True)
ws = wb.active
deliveries = {}
load_weights = {}

options = RouteOptions()
service = RouteService('dAGkCaPVqA3OcC5ws2Lfv8wsR9ro45oe')

for row in range(1, len(ws['A'])+1):
    if ('delivery' in str(ws['J'+str(row)].value)):
        if str(ws['A'+str(row)].value) in deliveries.keys():
            deliveries[str(ws['A'+str(row)].value)].append(str(ws['R'+str(row)].value))
            load_weights[float(ws['A'+str(row)].value)].append(float(ws['AK'+str(row)].value))
        else:
            deliveries[str(ws['A'+str(row)].value)] = [str(ws['R'+str(row)].value)]
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
  
def create_one_driver_graph():
    G = nx.DiGraph()
    drive_times = {}
    nodes = deliveries[driver]
    nodes.append(sleeman_location) # Add the origin destination
  
    for n in range(0, len(nodes)): # Might need to be len(nodes) + 1 instead, test and find out
        cycled_nodes = (nodes[-n:] + nodes[:-n])

        routeMatrix = calc_edges(cycled_nodes)
        for i in range(1, len(cycled_nodes)+1):
            #print cycled_nodes[0]+' ~ AND ~ '+cycled_nodes[i]
            import ipdb
            ipdb.set_trace()
            drive_times[(cycled_nodes[0], cycled_nodes[i])] = float(routeMatrix[i]) #KEY ERROR HERE! (not immutable?)
  
    G.add_nodes_from(nodes)
    G.add_edges_from(drive_times)
  
# NOT USED AS OF NOW, ONLY FOR REFERENCE PURPOSES
def create_all_driver_graphs():
    G = nx.DiGraph()
    drive_times = {}
    for driver in deliveries.keys():
        drive_times.clear()
        nodes = deliveries[driver]
        G.add_nodes_from(nodes)
        all_combs = combinations(nodes, 2) 
        for i in len(list(all_combs)):
            drive_times.add(list(all_combs)[i], calc_edges(list(all_combs)[i][0],list(all_combs)[i][1], 'dist'))
            G.add_edges_from(drive_times)
    return G
    # nx.shortest_path() investigation

def display_graph(G):
    nx.draw_circular(G,
        node_color=COLORS[0],
        node_size=2000,
        with_labels=True)


routeGraph = create_one_driver_graph()

display_graph(routeGraph)

################
# This section is TESTING for the weighted matrix outputs
################

"""Separates out the location list of a single truck"""
#Enter number for truck to get locations for
one_truck_locations = deliveries[driver]
one_truck_load_weights = load_weights[float(driver)]

print "Original Dist Matrix:"
route_matrix = service.routeMatrix(locations=one_truck_locations, oneToMany=True)
dist_matrix = route_matrix['distance']
print dist_matrix

print "Weighted Dist Matrix:"
weighted_dist_matrix = get_weighted_dist_matrix(one_truck_locations, one_truck_load_weights, 0.5)
print weighted_dist_matrix

print "Original Time Matrix:"
time_matrix = route_matrix['time']
print time_matrix

print "Weighted Time Matrix:"
weighted_time_matrix = get_weighted_time_matrix(one_truck_locations, one_truck_load_weights, 0.5)
print weighted_time_matrix