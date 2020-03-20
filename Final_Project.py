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

%matplotlib inline

wb = load_workbook('FILE_NAME', read_only=False, keep_vba=True)
ws = wb.active
deliveries = {}
load_weights = []

for row in range(1, len(ws['A'])+1):
	if ('delivery' in str(ws['J'+str(row)].value)):
		if ws['A'+str(row)].value in deliveries.keys():
			deliveries[ws['A'+str(row)].value].append(ws['R'+str(row)].value)
		else:
			deliveries[ws['A'+str(row)].value] = [ws['R'+str(row)].value]
		load_weights.append(ws['AK'+str(row)].value)


def get_weighted_dist_matrix(locations, loads, factor_weight):

    route_matrix = service.routeMatrix(locations=locations, allToAll=true)
    dist_matrix = route_matrix['distance']
    weighted_dist_matrix = dist_matrix

    i = 0;
    for load_weight in loads:
        weighted_dist_matrix[i] = dist_matrix[i]*load_weight*factor_weight
        ++i
    return weighted_dist_matrix


def get_weighted_time_matrix(locations, loads, factor_weight):

    route_matrix = service.routeMatrix(locations=locations, allToAll=true)

    time_matrix = route_matrix['time']
    weighted_time_matrix = time_matrix

    i = 0;
    for load_weight in loads:
        weighted_time_matrix[i] = time_matrix[i] * load_weight * factor_weight
        ++i
    return weighted_time_matrix


def calc_edge(addr_1, addr_2, factor=None):
	"""Calculate the edge weight based on certain factors"""

	service = RouteService('YOUR_MAPQUEST_DEVELOPER_API_KEY')
	location_list=[addr_1, addr_2]

	if factor is None or factor is 'dist':
		# MapQuest distance from addr_1 to addr_2
		routeMatrix = service.routeMatrix(locations=location_list, allToAll=True)
		return routeMatrix['distance'][0][1]

	elif factor is 'dist':
		return 0 # Placeholder 

	elif factor is 'time':
		return 0 # Placeholder
		# Return MapQuest time

	#elif factor is 'dist/time':
		# MapQuest time divided by distance

	#elif factor is 'fuel':
		# Mapquest distance from l1 to l2 
		# Taking weight into account for fuel efficiency
		# Use the weight_point_for_load() method
	
def weight_point_for_load(point):
  """Takes a delivery destination. Incorporates
  a load size based on a specified weighing factor. (changes the
  effective distance from this point to all others)"""

	load_factor_weight = 0.5 #experiment with different weights of this factor

	return weighted_loc
  
def create_individual_driver_graphs(nodes):
	G = nx.DiGraph()
	drive_times = {}
	for driver in deliveries.keys():
		drive_times.clear()
		nodes = deliveries[driver]
		G.add_nodes_from(nodes)
		all_combs = combinations(nodes, 2) 
		for i in len(list(all_combs)):
			drive_times.add(list(all_combs)[i], calc_edge(list(all_combs)[i][0],list(all_combs)[i][1]), 'dist')
			G.add_edges_from(drive_times)
	
	# nx.shortest_path() investigation
