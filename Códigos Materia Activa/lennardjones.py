import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import trange
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numba
from numba import njit, jit
matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\Desktop\\ffmpeg\\bin\\ffmpeg.exe"
writer = FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)
# Parametros y variables globales:
sigma=1
masa=1
epsilon=.1
sqrtN=15
Temperatura=0.01
Ttotal = 2
Transiente = 0
dt= 1e-3
velocitymagnitude = 50
D_r = .01
D_T = 0
frameskip = 1
softening = 0.1*sigma

radioc=2.5*sigma 
radioc2=radioc**2
N = sqrtN*sqrtN
L=np.sqrt((N*np.pi*sigma**2)/0.8) # Tamaño de la caja. Puedo poner int pq N es un cuadrado perfecto
sqrtdt = np.sqrt(dt)
Npasos = int(Ttotal/dt)
ratio = 1**(1/Npasos)
Ntransiente = int(Transiente/dt)
Nceldas = int(L/radioc)
delta = L/Nceldas
coefphi = np.sqrt(2*D_r)
coefpos = np.sqrt(2*D_T)
# Funciones:

@jit
def fuerzapar(p1, p2):
    dx=p1[0]-p2[0]
    dy=p1[1]-p2[1]
    if (dx>0.5*L):
        dx=dx-L
    if (dx<-0.5*L):
        dx=dx+L
    if (dy>0.5*L):
        dy=dy-L
    if (dy<-0.5*L):
        dy=dy+L  
    rad = dx**2 + dy**2
    if rad < radioc2:
        coef = 48*epsilon*((2*sigma**12)/(rad**7))
        fuerzax = coef*dx#48*epsilon*((2*sigma**12)/(rad**7))*moddif
        fuerzay = coef*dy
        return [fuerzax, fuerzay]
    else:
        return 0


def condicioninicial():
    global particles, velocidadinteractiva, phi
    velocidadinteractiva = np.sqrt(2*Temperatura/masa)*norm.rvs(size=(N,2))
    velocidadinteractiva[:, 0] = velocidadinteractiva[:,0] - np.mean(velocidadinteractiva[:,0])
    velocidadinteractiva[:, 1] = velocidadinteractiva[:,1] - np.mean(velocidadinteractiva[:,1])
    lin, espaciado = np.linspace(0, L, sqrtN, endpoint=False, retstep=True)
    particles[:, 0] = np.meshgrid(lin,lin)[0].flatten() + espaciado/2
    particles[:, 1] = np.meshgrid(lin,lin)[1].flatten() + espaciado/2
    return


def rotationarray(phi):
    global velocitymagnitude
    x = velocitymagnitude*np.cos(phi)
    y = velocitymagnitude*np.sin(phi)
    return np.array([x, y]).T


def termostato():
    global velocidadinteractiva
    vx = velocidadinteractiva[:,0]
    vy = velocidadinteractiva[:,1]
    Tcinetica = np.mean((masa/2)*(vx**2+vy**2))
    ajuste = np.sqrt(Temperatura/Tcinetica)
    velocidadinteractiva = velocidadinteractiva*ajuste
    return


def animable(i):
    global listaposiciones, scatter, ax, listaL
    ax.set_xlim((0, listaL[i]))
    ax.set_ylim((0, listaL[i]))
    tiempo = dt*i
    data = listaposiciones[i]
    scatter.set_offsets(data)
    tiempo_text.set_text("t = %.2f" % tiempo)
    return scatter, tiempo_text, ax


def crearlista():
    lista = []
    for i in range(Nceldas):
        lista.append([])
        for j in range(Nceldas):
            lista[i].append([])
    return lista
# Defino arreglo de listas enlazadas:

listaposiciones = []
listavelocidades = []
listaL = []
particles = np.zeros((N, 2), dtype="float64")
phi = np.random.uniform(low=-np.pi, high=np.pi, size=N)
velocidadinteractiva = np.zeros((N,2))
condicioninicial()

for t in trange(Ntransiente + Npasos):
    accel = np.zeros((N, 2))
    lista = crearlista()
    for particle in particles:
        n = int(particle[0]/delta) 
        m = int(particle[1]/delta)
        lista[n][m].append(particle)
    for i in range(N):
        particle = particles[i]
        n0 = int(particle[0]/delta)
        m0 = int(particle[1]/delta)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (n0 + dx) % Nceldas
                m = (m0 + dy) % Nceldas
                for otherparticle in lista[n][m]:
                    if not np.array_equal(otherparticle, particle):
                        accel[i] += np.array(fuerzapar(particle, otherparticle))/masa
    if t % 100 == 0:
        termostato()

    phi += np.random.normal(size=N)*coefphi*sqrtdt
    velocidadinteractiva = velocidadinteractiva + accel*dt
    newarray = particles + (velocidadinteractiva + rotationarray(phi))*dt #+ np.random.normal(size=(N, 2))*coefpos*sqrtdt# Truquito pq o sino mod actuaba raro.
    particles = np.mod(newarray, L) # Para mantener las condiciones de borde.
    particles *= ratio
    L *= ratio
    Nceldas = int(L/radioc)
    delta = L/Nceldas
    if t >= Ntransiente:
        if t%frameskip == 0:
            listaposiciones.append(particles.copy())
            listavelocidades.append(velocidadinteractiva.copy())
            listaL.append(L)

listaposiciones = np.array(listaposiciones)
listavelocidades = np.array(listavelocidades)

fig = plt.figure(figsize=(7,7))
fig.clf()
ax = plt.axes(xlim=(0,L),ylim=(0,L))
scatter = ax.scatter(listaposiciones[0,0], listaposiciones[0,1])
tiempo_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
anim = FuncAnimation(fig, animable, frames=int(Npasos/frameskip), interval=33)
#anim.save('anim.mp4', writer=writer) 
plt.show()
