import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

data = np.load("1024particles0.1gamma50velocity5time.npz")

positions = data["positions"]
boxsizes = data["boxsizes"]
parameters = data["parameters"]


voronoi_last_frame = Voronoi(positions[-1])


fig = voronoi_plot_2d(voronoi_last_frame, show_vertices = False)
plt.show()