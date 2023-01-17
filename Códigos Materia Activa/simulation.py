import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
from numba import njit
from tqdm import trange
import copy, cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\Desktop\\ffmpeg\\bin\\ffmpeg.exe"
#rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\OneDrive\\Escritorio\\ffmpeg-2023-01-01-git-62da0b4a74-essentials_build\\bin\\ffmpeg.exe"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=2800)

# Parametros y variables globales:

sqrtN=10
Ttotal = 5
Ttransient = 0 # Doesnt quite work
dt = 1e-3
Temperatura= 0.0
packing = .5
gammaexpansion = np.log(1)/Ttotal
mu = dt*0.1


# Parametros de potenciales:

tipo = int(input("Press 1 for Lennard Jones potential, 2 for Harmonic potential: "))
sigma=1
masa=1
epsilon=1
radiocorte = sigma*2**(1/6) # 2.5sigma era antes

# Parametros activos:

velocitymagnitude = 10
D_r = 1
D_T = 0.1 # DE MOMENTO NO HACE NADA

# Parámetro animación:

frameskip = 100 # Stores data every 10 frames.

# Funciones de otras cosas:

Nparticles = sqrtN*sqrtN
Nsteps = int(Ttotal/dt)
Ntransient = int(Ttransient/dt)
if tipo == 1:
    boxsize = sqrtN*(sigma/2)*np.sqrt(np.pi/packing)
    Ncells = max(int(boxsize/radiocorte), 1)
if tipo == 2:
    boxsize = sqrtN*(sigma)*np.sqrt(np.pi/2*packing)
    Ncells = max(int(boxsize/sigma), 1)
delta = boxsize/Ncells
ratio = 1 + gammaexpansion*dt
coefphi = np.sqrt(2*D_r*dt)
coefpos = np.sqrt(2*D_T*dt)
sqrtdt = np.sqrt(dt)

# Profiling:

# pr.enable()

# Funciones:

@njit
def distances(particle1, particle2, boxsize): # Could be done using modulus function for part1 - part2 + .5*boxsize and subtracting .5*boxsize at the end.
    """ 
    Computes the distance between two particles.
    Corrects for boundary conditions. Here, particles
    are represented by 1D arrays with their position
    vector coordinates. 
    """
    dP = particle1 - particle2 + .5*boxsize
    dP = dP % boxsize
    dP -= .5*boxsize
    dx, dy = dP
    r = np.linalg.norm(dP)
    return dx, dy, r

@njit
def arrayequal(particle1, particle2):
    """
    JIT version of np.array_equal. Returns True when p1 and p2
    are equal element-wise, meaning they are the same particle.
    """
    boolean = np.array_equal(particle1, particle2)
    return boolean

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
            coef = 4*epsilon*(12*(sigma**12)/(r**14) - (sigma**6)/(r**8))
    if type == 2:
        if r < sigma:
            coef = (epsilon/sigma)*(1/r-1/sigma)

    xforce = coef*dx
    yforce = coef*dy
    return xforce, yforce

def force(particle, boxsize, delta, cell_list, type, ncells=Ncells):
    """
    Computes the force felt by a particle subject
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
                if not arrayequal(particle, otherparticle):
                    xforce, yforce = pairwiseforce(particle, otherparticle, boxsize, type)
                    xaccel += xforce
                    yaccel += yforce
    return xaccel, yaccel

def update_phi(oldphi):
    """
    Updates the angles of the persistance velocity with random noise.
    """
    phi = oldphi + np.random.normal(size=Nparticles)*coefphi
    return phi

def fill_cell_list(particles, delta, emptylist):
    """
    Fills a 2D cell list of Ncells x Ncells, where each cell
    contains particles that lie within it. Could be very
    slow.
    """
    for particle in particles:
        nn = int(particle[0]//delta)
        mm = int(particle[1]//delta)
        emptylist[nn][mm].append(particle)

    return emptylist


def condicioninicial(Nparticles, temperatura, masa, boxsize):
    """
    Sets as the initial condition particles in a 2D mesh and random initial velocities.
    It also adds random angles to the persistance velocities (separately from thermal 
    velocities).
    """
    
    lin, espaciado = np.linspace(0, boxsize, sqrtN, endpoint=False, retstep=True)
    particles = np.zeros((Nparticles, 2), dtype="float64")
    particles[:, 0] = np.meshgrid(lin,lin)[0].flatten() + espaciado/2
    particles[:, 1] = np.meshgrid(lin,lin)[1].flatten() + espaciado/2
    phi = np.random.uniform(-np.pi, np.pi, Nparticles)
    return particles, phi

@njit
def persistancevelocity(phi, velocitymagnitude):
    """
    Calculates the velocity vector of all particles, which is determined 
    by the angles phi. This corresponds only to the velocity resulting from
    self propulsion. 
    """
    v = velocitymagnitude*np.column_stack((np.cos(phi), np.sin(phi)))
    return v

@njit
def update_positions(oldparticles, forcearray, phi, velocitymagnitude, dt):
    """
    Performs a numerical integration of the equations of motion based on previous
    positions, an array with the force felt by each particle and the angles of the
    self propulsion velocity.
    """
    particles = oldparticles + mu*forcearray + persistancevelocity(phi, velocitymagnitude)*dt
    return particles

@njit
def boundary_and_expand(weirdparticles, boxsize, ratio):
    """
    Applies the periodic boundary conditions for particles (named weirdparticles
    because they can be outside the box). Then, expands or contracts the space
    based on ratio.
    """
    particles = weirdparticles % boxsize
    particles *= ratio
    boxsize *= ratio
    if tipo == 1:
        Newcells = max(int(boxsize/radiocorte), 1)
    if tipo == 2:
        Newcells = max(int(boxsize/sigma), 1)
    delta = boxsize/Newcells
    return particles, boxsize, Newcells, delta


def create_cell_list(Ncells):
    """
    Creates a linked cell list to be filled later.
    """
    tocopy = []
    for i in range(Ncells):
        tocopy.append([])
        for j in range(Ncells):
            tocopy[i].append([])
    return tocopy


particles, phi = condicioninicial(Nparticles, Temperatura, masa, boxsize)
store_positions = np.zeros((Nsteps//frameskip, Nparticles, 2))
store_boxsize = np.zeros((Nsteps//frameskip))
k=0
tocopy = create_cell_list(Ncells)
expansionratio = 1
for t in trange(Nsteps + Ntransient):
    if t == Ntransient:
        expansionratio = ratio

    if expansionratio == 1: # Could be faster this way.
        cell_list = copy.deepcopy(tocopy)
    else:
        cell_list = create_cell_list(Ncells) # Could be optimized so it creates a cell_list each time Ncells changes.
    cell_list = fill_cell_list(particles, delta, cell_list)
    forces = np.zeros((Nparticles, 2))
    for i in range(Nparticles):
        particle = particles[i]
        forces[i] = force(particle, boxsize, delta, cell_list, tipo, ncells=Ncells)
    phi = update_phi(phi)
    weirdparticles = update_positions(particles, forces, phi, velocitymagnitude, dt)
    particles, boxsize, Ncells, delta = boundary_and_expand(weirdparticles, boxsize, ratio)

    if t >= Ntransient:
        if t % frameskip == 0:
            store_positions[k] = particles.copy()
            store_boxsize[k] = boxsize
            k+=1


# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# with open('test.txt', 'w+') as f:
#     f.write(s.getvalue())

if input("Animate? (y/n) ") == "y":
    def animable(i):
        global store_positions, scatter, ax, store_boxsize
        if gammaexpansion != 0:
            ax.set_xlim((0, store_boxsize[i]))
            ax.set_ylim((0, store_boxsize[i]))
            sqrtS = int(300/(sqrtN*ratio**(i*frameskip)))
            scatter.set_sizes(sqrtS*sqrtS*np.ones(Nparticles))
        tiempo = dt*i*frameskip
        data = store_positions[i]
        scatter.set_offsets(data)
        tiempo_text.set_text("t = %.2f" % tiempo)
        return scatter, tiempo_text, ax

    fig = plt.figure(figsize=(7,7))
    fig.clf()
    ax = plt.axes(xlim=(0,store_boxsize[0]),ylim=(0,store_boxsize[0]))
    marksize = int(300/sqrtN)*int(300/sqrtN)
    scatter = ax.scatter(store_positions[0,:,0], store_positions[0,:,1], s=marksize)
    tiempo_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    anim = FuncAnimation(fig, animable, frames=int(Nsteps/frameskip), interval=10)
    if input("Save animation? (y/n) ") == "y":
        if tipo == 1:
            anim.save('lennardjones%i.mp4' % Nparticles, writer=writer)
        if tipo == 2:
            anim.save('harmonic%i.mp4' % Nparticles, writer=writer)
    plt.show()

if input("Save data as .npz? (y/n) ") == "y":
    parameters = np.array([sigma, epsilon, radiocorte, sqrtN, packing, mu, D_r, D_T, gammaexpansion, tipo, frameskip, dt])
    np.savez_compressed("%iparticles%.1fgamma%ivelocity%itime" % (Nparticles, gammaexpansion, velocitymagnitude, Ttotal), positions=store_positions, boxsizes=store_boxsize, parameters=parameters)