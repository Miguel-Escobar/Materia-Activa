import numpy as np
from analysis_functions import *
from scipy.stats import linregress
import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile() # Descomentar lineas para utilizar.

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

- Oswald Ripening - fenomeno de aerosol condensado

- 

"""
Nmeasurements = 5
velocity = 10.0
Measureinterval = 10
time = 5
Densitydistributionpoints = 3
initialpf = 0.9
clusterings = []
maxclusters = []
density_fluctuations_avgn = []
density_fluctuations_varn = []

pr.enable() # Activas el profiler
for i in range(Nmeasurements):
    filename = "Códigos Materia Activa\\Termalizado y expandido\\Hertzian\\%.1finitialpf400particles%.1fvelocity%itime%i.npz" % (initialpf,velocity,time, i)
    data = np.load(filename)
    positions = data["positions"]
    boxsizes = data["boxsizes"]
    parameters = data["parameters"]
    sigma = parameters[0]
    clustering, maxcluster = clustering_and_maxcluster(positions, boxsizes, sigma/2, Measureinterval)
    avg_n_vs_time, var_vs_time = density_fluctuations_vs_time(positions, boxsizes, 10, Densitydistributionpoints)
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


pr.disable() # Desactivas el profiler

s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
with open('test.txt', 'w+') as f: # Guardas los resultados en un .txt
    f.write(s.getvalue())


clusterings = np.array(clusterings)
maxclusters = np.array(maxclusters)
Nsteps = len(boxsizes)
dt = parameters[-1]
frameskip = parameters[-2]
Nparticles = parameters[3]*parameters[3]
clusteringavg = np.mean(clusterings, axis=0)
clusteringstderr = np.std(clusterings, axis=0)/np.sqrt(Nmeasurements)
maxclusteravg = np.mean(maxclusters, axis=0)
maxclusterstderr = np.std(maxclusters, axis=0)/np.sqrt(Nmeasurements)
time = np.arange(Nsteps)*dt*frameskip
exponent_vs_time = np.array(exponent_vs_time)
packing_fractions = Nparticles*(np.pi*(sigma/2)**2)/(boxsizes[[i*Measureinterval for i in range(int(Nsteps/Measureinterval))]]**2)
plottable_time = time[[i*Measureinterval for i in range(int(Nsteps/Measureinterval))]]
np.savez("analyzed_data%.1fvelocity" % velocity, clusterings = clusterings, maxclusters=maxclusters, time=time, plottable_time=plottable_time, \
        clusteringavg=clusteringavg, clusteringstderr=clusteringstderr, maxclusteravg=maxclusteravg, maxclusterstderr=maxclusterstderr,\
        packing_fractions=packing_fractions)

