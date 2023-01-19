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
from analysis_functions import *
#rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\Desktop\\ffmpeg\\bin\\ffmpeg.exe"
rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\OneDrive\\Escritorio\\ffmpeg-2023-01-01-git-62da0b4a74-essentials_build\\bin\\ffmpeg.exe"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=2800)
"""
Datos interesantes:

Sistemas sin expansión:
- LISTO Coeficiente de clustering en funcion del tiempo (voronoi o con componentes conexas). Para ambos necesito encontrar un radio crítico.
- Fittear cte*(1-exp(-t/Tau)) donde Tau es tiempo de relajación (llegar a estado estacionario).
- LISTO Tamaño cluster más grande en función del tiempo.

Sistemas en expansión:
- Medir lo mismo que sin expansión para caso de sistema sin termalizar y de sistema termalizado.

Para hacer estadística:
- 400 particulas +- 20 veces?
- Puedo medir más o menos cada 100 o 10 iteraciones

Identificar qué es lo que pasa cuando explota la simulación. ¿Cutoff a la velocidad de las partículas?

Pasar la custion al error estandar

Giant number fluctuations - Plotear var vs el promedio achicando cajas o bien var vs tamaño de mis cajitas. Graficar en log.
La idea es sacar un fit de linea recta. Más o menos 5 puntos. Puedo graficar el exponente en función del tiempo.

"""


clusterings = []
maxclusters = []

for i in range(10):
    filename = "No termalizado ni expandido\\Hertzian\\400particles50velocity10time%i.npz" % i
    data = np.load(filename)
    positions = data["positions"]
    boxsizes = data["boxsizes"]
    parameters = data["parameters"]
    sigma = parameters[0]
    clustering, maxcluster = clustering_and_maxcluster(positions, boxsizes, sigma/2, 10)
    clusterings.append(clustering)
    maxclusters.append(maxcluster)
clusterings = np.array(clusterings)
maxclusters = np.array(maxclusters)
Nsteps = len(clusterings[0, :])
dt = parameters[-1]
fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
clusteringavg = np.mean(clusterings, axis=0)
clusteringstd = np.std(clustering, axis=0)
maxclusteravg = np.mean(maxclusters, axis=0)
maxclusterstd = np.std(maxclusters, axis=0)
time = np.arange(Nsteps)*dt
ax1.errorbar(time, clusteringavg, clusteringstd)
ax2.errorbar(time, maxclusteravg, maxclusterstd)
ax1.set_xlabel("Time")
ax2.set_xlabel("Time")
ax1.set_ylabel(r"$\langle c_n(t) \rangle$")
ax2.set_ylabel(r"Largest Cluster Size")
fig.tight_layout()
fig.show()


    


    


