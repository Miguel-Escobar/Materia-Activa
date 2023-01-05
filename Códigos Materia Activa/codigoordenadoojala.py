import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
from numba import njit
from tqdm import trange
import copy
rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\Desktop\\ffmpeg\\bin\\ffmpeg.exe"
writer = FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)

# Parametros y variables globales:

sqrtN=2
Ttotal = 2
Ttransient = 0
dt = 1e-3
Temperatura=0.1
packing = .8
timesexpansion = 1


# Parametros de potenciales:

tipo = int(input("Press 1 for Lennard Jones potential, 2 for Harmonic potential: "))
sigma=1
masa=1
epsilon=1
radiocorte = 2.5*sigma

# Parametros activos:

velocitymagnitude = 1
D_r = 0.1
D_T = 0.1

# Parámetro animación:

frameskip = 10

# Funciones de otras cosas:

boxsize = sqrtN*sigma*np.sqrt(np.pi/packing)
Nparticles = sqrtN*sqrtN
Nsteps = int(Ttotal/dt)
Ntransient = int(Ttransient/dt)
if tipo == 1:
    Ncells = int(boxsize/radiocorte)
if tipo == 2:
    Ncells = int(boxsize/sigma)
delta = boxsize/Ncells
ratio = timesexpansion**(1/Nsteps)
coefphi = np.sqrt(2*D_r)
coefpos = np.sqrt(2*D_T)
sqrtdt = np.sqrt(dt)

# Funciones:

@njit
def distances(particle1, particle2, boxsize): # Could be done using modulus function for part1 - part2 + .5*boxsize and subtracting .5*boxsize at the end.
    """ 
    Computes the distance between two particles.
    Corrects for boundary conditions. Here, particles
    are represented by 1D arrays with their position
    vector coordinates. 
    """
    dx, dy = particle1 - particle2
    if (dx>0.5*boxsize):
        dx=dx-boxsize
    if (dx<-0.5*boxsize):
        dx=dx+boxsize
    if (dy>0.5*boxsize):
        dy=dy-boxsize
    if (dy<-0.5*boxsize):
        dy=dy+boxsize
    r = np.sqrt(dx**2 + dy**2)
    return dx, dy, r

@njit
def pairwiseforce(particle1, particle2, boxsize, type=tipo):
    """
    Computes the pairwise force between two particles, correcting
    for boundary conditions. Particles are 1D arrays storing position
    vector coordinates.
    """
    dx, dy, r = distances(particle1, particle2, boxsize)
    coef = 0
    if type == 1:
        if r < radiocorte:
            coef = 48*epsilon*((sigma**12)/(r**14))
    if type == 2:
        if r < sigma:
            coef = (epsilon/sigma)*(1/r-1/sigma)

    xforce = coef*dx
    yforce = coef*dy
    return xforce, yforce

@njit
def acceleration(particle, boxsize, delta, cell_list, type=tipo):
    """
    Computes the acceleration felt by a particle subject
    to a pairwise interaction potential. Particles is a 2
    dimensional array where each row is a set of coordinates
    for a particle. Cell_list is a 2D python list which stores
    particles in their respective cells. Delta is the cell size.
    """
    xaccel = 0.0
    yaccel = 0.0
    n = int(particle[0]/delta)
    m = int(particle[1]/delta)
    for otherparticle in cell_list[n][m]:
        if not np.array_equal(particle, otherparticle):
            xforce, yforce = pairwiseforce(particle, otherparticle, boxsize)
            xaccel += xforce/masa
            yaccel += yforce/masa
    return xaccel, yaccel

@njit
def fill_cell_list(particles, delta, emptylist):
    for particle in particles:
        n = particle[0]//delta
        m = particle[1]//delta
        emptylist[n][m].append(particle)
    return emptylist


velocities = np.ones((Nparticles,2))
phi = np.random.uniform(low=0.0, high=1.0, size=(Nparticles, 2))

def new_velocities(oldvelocities, oldphi):
    phi += np.random.normal(size=Nparticles)*D_r



tocopy = []
for i in range(Ncells):
    tocopy.append([])
    for j in range(Ncells):
        tocopy[i].append([])


