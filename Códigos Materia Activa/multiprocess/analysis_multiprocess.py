import numpy as np
from analysis_functions import *
import concurrent.futures as concfut
from scipy.stats import linregress
import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile() # Descomentar lineas para utilizar.

"""
NO FUNCIONA AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

"""
Nmeasurements = 2
velocity = 1.0
Measureinterval = 10
Densitydistributionpoints = 3
clusterings = []
maxclusters = []
density_fluctuations_avgn = []
density_fluctuations_varn = []

def analyze(inputlist):
    positions, boxsizes, sigma = inputlist
    clustering, maxcluster = clustering_and_maxcluster(positions, boxsizes, sigma/2, Measureinterval)
    avg_n_vs_time, var_vs_time = density_fluctuations_vs_time(positions, boxsizes, 10, Densitydistributionpoints)
    density_fluctuations_avgn.append(avg_n_vs_time)
    density_fluctuations_varn.append(var_vs_time)
    clusterings.append(clustering)
    maxclusters.append(maxcluster)
    return

futures = []
with concfut.ProcessPoolExecutor() as executor:
    for i in range(Nmeasurements):
        filename = "No termalizado ni expandido\\Hertzian\\400particles%.1fvelocity10time%i.npz" % (velocity, i)
        data = np.load(filename)
        positions = data["positions"]
        boxsizes = data["boxsizes"]
        parameters = data["parameters"]
        sigma = parameters[0]
        futures.append(executor.submit(analyze, [positions, boxsizes, sigma]))

    for future in concfut.as_completed(futures):
        print(future.result())

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

""" np.savez("analyzed_data%.1fvelocity" % velocity, clusterings = clusterings, maxclusters=maxclusters, time=time, plottable_time=plottable_time, \
        clusteringavg=clusteringavg, clusteringstderr=clusteringstderr, maxclusteravg=maxclusteravg, maxclusterstderr=maxclusterstderr,\
        packing_fractions=packing_fractions) """

