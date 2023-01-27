import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
rcParams.update({'font.size': 16})
kindofanalysis = input("Histieresis map or Relaxation time (hist/rel) ")

def fittable_relaxation_func(time, relaxtime, constant):
    return constant*(1-np.exp(-time/relaxtime))


if kindofanalysis == "hist":
    dataexpanded = np.load("Códigos Materia Activa\\No termalizado si expandido\\Hertzian\\analyzed_data_50velocity.npz")
    datacontracted = np.load("Códigos Materia Activa\\No termalizado si contraido\\Hertzian\\analyzed_data_50velocity.npz")
    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.errorbar(dataexpanded["packing_fractions"], dataexpanded["clusteringavg"], dataexpanded["clusteringstderr"], label="Expansión")
    ax1.errorbar(datacontracted["packing_fractions"], datacontracted["clusteringavg"], datacontracted["clusteringstderr"], label="Contracción")
    ax2.errorbar(dataexpanded["packing_fractions"], dataexpanded["maxclusteravg"], dataexpanded["maxclusterstderr"], label="Expansión")
    ax2.errorbar(datacontracted["packing_fractions"], datacontracted["maxclusteravg"], datacontracted["maxclusterstderr"], label="Contracción")
    ax1.set_xlabel("$\phi$")
    ax2.set_xlabel("$\phi$")
    ax1.set_ylabel(r"$\langle c_n(t) \rangle$")
    ax2.set_ylabel(r"Largest Cluster Size")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.show()

elif kindofanalysis == "rel":
    dataconstant = np.load("Códigos Materia Activa\\No termalizado ni expandido\\Hertzian\\analyzed_data10.0velocity0.1initialpf.npz")
    params = curve_fit(fittable_relaxation_func, dataconstant["plottable_time"], dataconstant["clusteringavg"], p0=[0.4,10])
    print("The relaxation time is: %.3f" % params[0][0])
    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.errorbar(dataconstant["plottable_time"], dataconstant["clusteringavg"], dataconstant["clusteringstderr"])
    ax2.errorbar(dataconstant["plottable_time"], dataconstant["maxclusteravg"], dataconstant["maxclusterstderr"])
    ax1.set_xlabel("Time")
    ax2.set_xlabel("Time")
    ax1.set_ylabel(r"$\langle c_n(t) \rangle$")
    ax2.set_ylabel(r"Largest Cluster Size")
    fig.tight_layout()
    fig.show()
elif kindofanalysis == "secretposter":
    # dataexpanded = np.load("Códigos Materia Activa\\Termalizado y expandido\\Hertzian\\analyzed_data10.0velocity5time.npz")
    # datacontracted = np.load("Códigos Materia Activa\\Termalizado y contraido\\Hertzian\\analyzed_data10.0velocity5time.npz")
    dataexpanded = np.load("Códigos Materia Activa\\No termalizado si expandido\\Hertzian\\analyzed_data_50velocity.npz")
    datacontracted = np.load("Códigos Materia Activa\\No termalizado si contraido\\Hertzian\\analyzed_data_50velocity.npz")
    fig = plt.figure(figsize=(7.7, 5))
    fig.clf()
    ax1 = fig.add_subplot(111)
    ax1.errorbar(dataexpanded["packing_fractions"], dataexpanded["maxclusteravg"], dataexpanded["maxclusterstderr"], label="Expansion", color="firebrick")
    ax1.errorbar(datacontracted["packing_fractions"], datacontracted["maxclusteravg"], datacontracted["maxclusterstderr"], label="Contraction", color="gray")
    ax1.set_xlabel("Packing Fraction")
    ax1.set_ylabel(r"Largest Cluster Size")
    ax1.legend()
    fig.tight_layout()
    fig.savefig("fasthystmap.svg", transparent=True)
    fig.show()
# elif kindofanalysis == "madness":
#     filename = "Códigos Materia Activa\\Termalizado y contraido\\Hertzian\\0.1initialpf400particles10.0velocity5time0.npz"

#     data = np.load(filename)
#     store_positions = data["positions"]
#     store_boxsize = data["boxsizes"]
#     parameters = data["parameters"]
#     sqrtN = parameters[3]
#     gammaexpansion = parameters[8]
#     dt = parameters[-1]
#     ratio = (1 + gammaexpansion*dt)
#     frameskip = parameters[-2]
#     tipo = parameters[-3]
#     Nsteps = len(store_boxsize)
#     Nparticles = len(store_positions[0, :, 0])