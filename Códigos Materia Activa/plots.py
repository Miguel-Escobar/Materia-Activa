import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
kindofanalysis = input("Histieresis map or Relaxation time (hist/rel) ")

def fittable_relaxation_func(time, relaxtime, constant):
    return constant*(1-np.exp(-time/relaxtime))


if kindofanalysis == "hist":
    dataexpanded = np.load("Códigos Materia Activa\\Termalizado y expandido\\Hertzian\\analyzed_data10.0velocity5time.npz")
    datacontracted = np.load("Códigos Materia Activa\\Termalizado y contraido\\Hertzian\\analyzed_data10.0velocity5time.npz")
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
