import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d

data = np.load("1024particles0.1gamma50velocity5time.npz")

positions = data["positions"]
boxsizes = data["boxsizes"]
parameters = data["parameters"]

def voronoi_volumes(v):
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



voronoi_last_frame = Voronoi(positions[-1])

fig = voronoi_plot_2d(voronoi_last_frame, show_vertices = False)
plt.show()


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