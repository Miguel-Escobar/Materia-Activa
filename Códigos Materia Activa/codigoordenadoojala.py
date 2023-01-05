import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
from numba import njit
from tqdm import trange
import copy
rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\OneDrive\\Escritorio\\ffmpeg-2023-01-01-git-62da0b4a74-essentials_build\\bin\\ffmpeg.exe"
writer = FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)

# Parametros y variables globales:

sqrtN=10
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
coefphi = np.sqrt(2*D_r*dt)
coefpos = np.sqrt(2*D_T*dt)
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
def pairwiseforce(particle1, particle2, boxsize, type):
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

def acceleration(particle, boxsize, delta, cell_list, type, ncells=Ncells):
    """
    Computes the acceleration felt by a particle subject
    to a pairwise interaction potential. Particles is a 2
    dimensional array where each row is a set of coordinates
    for a particle. Cell_list is a 2D python list which stores
    particles in their respective cells. Delta is the cell size.
    """
    xaccel = 0.0
    yaccel = 0.0
    n0 = int(particle[0]/delta)
    m0 = int(particle[1]/delta)
    for delx in (-1, 0, 1):
        for dely in (-1, 0 , 1):
            n = int((n0 + delx) % ncells)
            m = int((m0 + dely) % ncells)
            for otherparticle in cell_list[n][m]:
                if not np.array_equal(particle, otherparticle):
                    xforce, yforce = pairwiseforce(particle, otherparticle, boxsize, type)
                    xaccel += xforce/masa
                    yaccel += yforce/masa
    return xaccel, yaccel

def update_velocities(oldvelocities, oldphi, accelarray, dt):
    """
    Updates the velocities resulting from pairwise interaction (assuming medium with
    no viscocity) and updates the angles of the persistance velocity with random noise.
    """
    phi = oldphi + np.random.normal(size=Nparticles)*coefphi
    velocities = oldvelocities + accelarray*dt 
    return phi, velocities

def fill_cell_list(particles, delta, emptylist):
    """
    Fills a 2D cell list of Ncells x Ncells, where each cell
    contains particles that lie within it. Could be very
    slow.
    """
    for particle in particles:
        n = int(particle[0]//delta)
        m = int(particle[1]//delta)
        emptylist[n][m].append(particle)
    return emptylist


def condicioninicial(Nparticles, temperatura, masa, boxsize):
    velocities = np.sqrt(2*temperatura/masa)*np.random.normal(size=(Nparticles,2))
    velocities[:, 0] = velocities[:,0] - np.mean(velocities[:,0])
    velocities[:, 1] = velocities[:,1] - np.mean(velocities[:,1])
    lin, espaciado = np.linspace(0, boxsize, sqrtN, endpoint=False, retstep=True)
    particles = np.zeros((Nparticles, 2), dtype="float64")
    particles[:, 0] = np.meshgrid(lin,lin)[0].flatten() + espaciado/2
    particles[:, 1] = np.meshgrid(lin,lin)[1].flatten() + espaciado/2
    phi = np.random.uniform(-np.pi, np.pi, Nparticles)
    return particles, velocities, phi

@njit
def persistancevelocity(phi, velocitymagnitude):
    v = velocitymagnitude*np.column_stack((np.cos(phi), np.sin(phi)))
    return v

@njit
def update_positions(oldparticles, velocities, phi, velocitymagnitude, dt):
    particles = oldparticles + (velocities + persistancevelocity(phi, velocitymagnitude))*dt
    return particles

@njit
def boundary_and_expand(weirdparticles, boxsize, ratio):
    particles = weirdparticles % boxsize
    particles *= ratio
    boxsize *= ratio
    return particles, boxsize



tocopy = []
for i in range(Ncells):
    tocopy.append([])
    for j in range(Ncells):
        tocopy[i].append([])


particles, velocities, phi = condicioninicial(Nparticles, Temperatura, masa, boxsize)

store_positions = np.zeros((Nsteps//frameskip, Nparticles, 2))
store_boxsize = np.zeros((Nsteps//frameskip))
k=0

for t in trange(Nsteps + Ntransient):
    cell_list = copy.deepcopy(tocopy)
    cell_list = fill_cell_list(particles, delta, cell_list)
    accelerations = np.zeros((Nparticles, 2))
    for i in range(Nparticles):
        particle = particles[i]
        accelerations[i] = acceleration(particle, boxsize, delta, cell_list, tipo)
    phi, velocities = update_velocities(velocities, phi, accelerations, dt)
    weirdparticles = update_positions(particles, velocities, phi, velocitymagnitude, dt)
    particles, boxsize = boundary_and_expand(weirdparticles, boxsize, ratio)
    if tipo == 1:
        Ncells = int(boxsize/radiocorte)
    if tipo == 2:
        Ncells = int(boxsize/sigma)
    if t >= Ntransient:
        if t%frameskip == 0:
            store_positions[k] = particles.copy()
            store_boxsize[k] = boxsize
            k+=1



def animable(i):
    global store_positions, scatter, ax, store_boxsize
    # ax.set_xlim((0, store_boxsize[i]))
    # ax.set_ylim((0, store_boxsize[i]))
    tiempo = dt*i*frameskip
    data = store_positions[i]
    scatter.set_offsets(data)
    tiempo_text.set_text("t = %.2f" % tiempo)
    return scatter, tiempo_text, ax


fig = plt.figure(figsize=(7,7))
fig.clf()
ax = plt.axes(xlim=(0,boxsize),ylim=(0,boxsize))
scatter = ax.scatter(store_positions[0,0], store_positions[0,1])
tiempo_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
anim = FuncAnimation(fig, animable, frames=int(Nsteps/frameskip), interval=33)
#anim.save('anim.mp4', writer=writer) 
plt.show()












