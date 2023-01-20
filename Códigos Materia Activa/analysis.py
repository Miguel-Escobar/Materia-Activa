import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
import networkx as nx
from tess import Container
from numba import njit
from tqdm import trange
from analysis_functions import *
from scipy.stats import linregress
#rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\Desktop\\ffmpeg\\bin\\ffmpeg.exe"
rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\OneDrive\\Escritorio\\ffmpeg-2023-01-01-git-62da0b4a74-essentials_build\\bin\\ffmpeg.exe"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=2800)
"""
Datos interesantes:

Sistemas sin expansión:
- LISTO Coeficiente de clustering en funcion del tiempo (voronoi o con componentes conexas). Para ambos necesito encontrar un radio crítico.
- YA NO VA Fittear cte*(1-exp(-t/Tau)) donde Tau es tiempo de relajación (llegar a estado estacionario).
- LISTO Tamaño cluster más grande en función del tiempo.

Sistemas en expansión:
- Medir lo mismo que sin expansión para caso de sistema sin termalizar y de sistema termalizado.

- Giant number fluctuations - Plotear var vs el promedio achicando cajas o bien var vs tamaño de mis cajitas. Graficar en log.
La idea es sacar un fit de linea recta. Más o menos 5 puntos. Puedo graficar el exponente en función del tiempo.

- Plotear en función de la packing fraction.

"""
Nmeasurements = 5

clusterings = []
maxclusters = []
density_fluctuations_avgn = []
density_fluctuations_varn = []

for i in range(Nmeasurements):
    filename = "No termalizado si expandido\\Hertzian\\400particles50velocity10time%i.npz" % i
    data = np.load(filename)
    positions = data["positions"]
    boxsizes = data["boxsizes"]
    parameters = data["parameters"]
    sigma = parameters[0]
    clustering, maxcluster = clustering_and_maxcluster(positions, boxsizes, sigma/2, 10)
    avg_n_vs_time, var_vs_time = density_fluctuations_vs_time(positions, boxsizes, 10, 3)
    density_fluctuations_avgn.append(avg_n_vs_time)
    density_fluctuations_varn.append(var_vs_time)
    clusterings.append(clustering)
    maxclusters.append(maxcluster)
density_fluctuations_avgn = np.array(density_fluctuations_avgn)
density_fluctuations_varn = np.array(density_fluctuations_varn)
fittable_avgn = np.log(np.mean(density_fluctuations_avgn, axis=0))
fittable_var = np.log(np.mean(density_fluctuations_varn, axis=0))
exponent_vs_time = []
for time in range(len(fittable_var[:,0])-1):
    time += 1
    fiteo = linregress(fittable_avgn[time], fittable_var[time])
    exponent_vs_time.append(fiteo.slope)



exponent_vs_time = np.array(exponent_vs_time)
clusterings = np.array(clusterings)
maxclusters = np.array(maxclusters)
Nsteps = len(clusterings[0, :])
dt = parameters[-1]
Nparticles = parameters[3]*parameters[3]
clusteringavg = np.mean(clusterings, axis=0)
clusteringstderr = np.std(clusterings, axis=0)/np.sqrt(Nmeasurements)
maxclusteravg = np.mean(maxclusters, axis=0)
maxclusterstderr = np.std(maxclusters, axis=0)/np.sqrt(Nmeasurements)
time = np.arange(Nsteps)*dt
packingfractions = Nparticles*(np.pi*(sigma/2)**2)/boxsizes




fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.errorbar(time, clusteringavg, clusteringstderr)
ax2.errorbar(time, maxclusteravg, maxclusterstderr)
ax1.set_xlabel("Time")
ax2.set_xlabel("Time")
ax1.set_ylabel(r"$\langle c_n(t) \rangle$")
ax2.set_ylabel(r"Largest Cluster Size")
fig.tight_layout()
fig.show()


    


    


