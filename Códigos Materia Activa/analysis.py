import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
import networkx as nx
from tess import Container
from numba import njit
from tqdm import trange
"""
Datos interesantes:

Sistemas sin expansión:
- Coeficiente de clustering en funcion del tiempo (voronoi o con componentes conexas). Para ambos necesito encontrar un radio crítico.
- Fittear cte*(1-exp(-t/Tau)) donde Tau es tiempo de relajación (llegar a estado estacionario).
- Tamaño cluster más grande en función del tiempo.

Sistemas en expansión:
- Medir lo mismo que sin expansión para caso de sistema sin termalizar y de sistema termalizado.


Para hacer estadística:
- 400 particulas +- 20 veces?
- Puedo medir más o menos cada 100 o 10 iteraciones
"""

data = np.load("1024particles0.1gamma50velocity5time.npz")

positions = data["positions"]
boxsizes = data["boxsizes"]
parameters = data["parameters"]

@njit
def distances(particle1, particle2, boxsize): # Could be done using modulus function for part1 - part2 + .5*boxsize and subtracting .5*boxsize at the end.
    """ 
    Computes the distance between two particles.
    Corrects for boundary conditions. Here, particles
    are represented by 1D arrays with their position
    vector coordinates. 
    """
    dP = particle1 - particle2 + .5*boxsize
    dP = dP % boxsize
    dP -= .5*boxsize
    r = np.linalg.norm(dP)
    return r

def average_voronoi_volumes(v):
    vol = np.zeros(v.npoints)
    npolygons = 0
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            pass
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
            npolygons += 1
    ratio = np.sum(vol)/npolygons
    return vol, ratio

def clustering_coefficient(points, boxsize, critical_radius):
    embedded_positions = np.hstack((points, .5*np.ones((len(points[:,0]), 1))))
    voronoi_cells_container = Container(embedded_positions, limits=(boxsize, boxsize, 1), periodic=(True, True, False))
    N = len(voronoi_cells_container)
    critical_area = np.pi*critical_radius**2
    clustered_counter = 0
    for cell in voronoi_cells_container:
        if cell.volume() <= critical_area:
            clustered_counter += 1
    return clustered_counter/N




def get_distance_based_neighbourhood_graph(particles, neighbourDistance, boxsize):
    # Creates graph whose connected components are the clusters, defined by two particles being within neighbourhood distance
    #Initialise the graph
    G =  nx.Graph()
    Nparticles = len(particles[:,0])

    for nodeIdx in range(Nparticles):
        G.add_node(nodeIdx)

    for node1 in range(Nparticles):
        for node2 in range(node1):
            if(np.abs(distances(particles[node1], particles[node2], boxsize)) < neighbourDistance):
                G.add_edge(node1, node2)
                G.add_edge(node2, node1)
    return G

def max_cluster_size(graph):
    connectedComponentsSize = [len(comp) for comp in nx.connected_components(graph)]
    return np.max(connectedComponentsSize)

def analyze_simulation(positions, boxsizes, critical_radius, measure_interval):
    Nparticles = len(positions[0,:,0])
    sample_number = int(len(positions[:,0,0])/measure_interval)
    clustering_coefficients = np.zeros(sample_number)
    max_cluster_ratios = np.zeros(sample_number)
    for i in trange(sample_number):
        neighbourhood_graph = get_distance_based_neighbourhood_graph(positions[i*measure_interval], critical_radius, boxsizes[i*measure_interval])
        max_cluster_ratios[i] = max_cluster_size(neighbourhood_graph)/Nparticles
        clustering_coefficients[i] = clustering_coefficient(positions[i*measure_interval], boxsizes[i*measure_interval], critical_radius)
    return clustering_coefficients, max_cluster_ratios

    


    


