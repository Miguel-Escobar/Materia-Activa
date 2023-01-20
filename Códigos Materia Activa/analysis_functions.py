import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
import networkx as nx
from tess import Container
from numba import njit
from tqdm import trange
import pdb
#rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\Desktop\\ffmpeg\\bin\\ffmpeg.exe"
rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\OneDrive\\Escritorio\\ffmpeg-2023-01-01-git-62da0b4a74-essentials_build\\bin\\ffmpeg.exe"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=2800)

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

def clustering_and_maxcluster(positions, boxsizes, critical_radius, measure_interval):
    Nparticles = len(positions[0,:,0])
    sample_number = int(len(positions[:,0,0])/measure_interval)
    clustering_coefficients = np.zeros(sample_number)
    max_cluster_ratios = np.zeros(sample_number)
    for i in trange(sample_number):
        neighbourhood_graph = get_distance_based_neighbourhood_graph(positions[i*measure_interval], 2*critical_radius, boxsizes[i*measure_interval])
        max_cluster_ratios[i] = max_cluster_size(neighbourhood_graph)/Nparticles
        clustering_coefficients[i] = clustering_coefficient(positions[i*measure_interval], boxsizes[i*measure_interval], critical_radius)
    return clustering_coefficients, max_cluster_ratios

def animate(filename):
    data = np.load(filename)
    store_positions = data["positions"]
    store_boxsize = data["boxsizes"]
    parameters = data["parameters"]
    sqrtN = parameters[3]
    gammaexpansion = parameters[8]
    dt = parameters[-1]
    ratio = (1 + gammaexpansion*dt)
    frameskip = parameters[-2]
    tipo = parameters[-3]
    Nsteps = len(store_boxsize)
    Nparticles = len(store_positions[0, :, 0])
    
    def animable(i, scatter, ax, store_positions, store_boxsize, tiempo_text, marksize):
        if gammaexpansion != 0:
            ax.set_xlim((0, store_boxsize[i]))
            ax.set_ylim((0, store_boxsize[i]))
            marksize = marksize/(ratio**(2*i*frameskip))
            scatter.set_sizes(marksize*np.ones(Nparticles))
        tiempo = dt*i*frameskip
        data = store_positions[i]
        scatter.set_offsets(data)
        tiempo_text.set_text("t = %.2f" % tiempo)
        return scatter, tiempo_text, ax

    figanimable = plt.figure(figsize=(7,7))
    figanimable.clf()
    ax = figanimable.add_subplot(xlim=(0,store_boxsize[0]),ylim=(0,store_boxsize[0]))
    marksize = int(300/sqrtN)*int(300/sqrtN)
    if gammaexpansion <= 0:
        marksize = marksize/((1 - gammaexpansion*dt)**(2*Nsteps*frameskip))
    
    scatter = ax.scatter(store_positions[0,:,0], store_positions[0,:,1], s=marksize)
    tiempo_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    anim = FuncAnimation(figanimable, animable, fargs=(scatter, ax, store_positions, store_boxsize, tiempo_text, marksize) ,frames=Nsteps, interval=10)
    if input("Save animation? (y/n) ") == "y":
        if tipo == 1:
            anim.save('lennardjones%i.mp4' % Nparticles, writer=writer)
        if tipo == 2:
            anim.save('harmonic%i.mp4' % Nparticles, writer=writer)

def fill_cell_list(particles, delta, emptylist):
    """
    Fills a 2D cell list of Ncells x Ncells, where each cell
    contains particles that lie within it. Could be very
    slow.
    """
    for particle in particles:
        nn = int(particle[0]//delta)
        mm = int(particle[1]//delta)
        emptylist[nn][mm].append(particle)

    return emptylist

def create_cell_list(sqrtNcells):
    """
    Creates a linked cell list to be filled later.
    """
    tocopy = []
    for i in range(sqrtNcells):
        tocopy.append([])
        for j in range(sqrtNcells):
            tocopy[i].append([])
    return tocopy

def density_fluctuations(particles, boxsize, Npoints):
    Nparticles = len(particles[:,0])
    avg_particles_in_cells = []
    variance = []
    for sqrtNcells in [int(2**(n+1)) for n in range(Npoints)]:
        Ncells = sqrtNcells*sqrtNcells
        avg = Nparticles/Ncells
        cellsize = boxsize/sqrtNcells
        cell_list = fill_cell_list(particles, cellsize, create_cell_list(sqrtNcells))
        unbiased_variance_estimator = 0
        for xindex in range(sqrtNcells):
            for yindex in range(sqrtNcells):
                Nparticles = len(cell_list[xindex][yindex])
                unbiased_variance_estimator += (Nparticles - avg)**2
        unbiased_variance_estimator *= 1/(Ncells-1)
        variance.append(unbiased_variance_estimator)
        avg_particles_in_cells.append(avg)
    return np.array(avg_particles_in_cells), np.array(variance)

def density_fluctuations_vs_time(positions, boxsizes, Ntimes, Npoints):
    timeinterval = int(len(positions[:,0,0])/Ntimes)
    avg_n_arrays = []
    var_arrays = []
    for i in range(Ntimes):
        returnedtuple = density_fluctuations(positions[i*timeinterval], boxsizes[i*timeinterval], Npoints)
        avg_n_arrays.append(returnedtuple[0])
        var_arrays.append(returnedtuple[1])
    return np.array(avg_n_arrays), np.array(var_arrays)








    