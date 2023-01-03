import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import trange
from matplotlib.animation import FuncAnimation
import pdb

# Parametros y variables globales:
sigma=1
masa=1
epsilon=1
sqrtN=25
Temperatura=0.1
Ttotal = 2
Transiente = 0
dt= 1e-3
velocitymagnitude = 3
D_r = 1
D_T = .1
frameskip = 10

radioc=2.5*sigma 
radioc2=radioc**2
N = sqrtN*sqrtN
L=int(np.sqrt(N*sigma**2)) # Tamaño de la caja. Puedo poner int pq N es un cuadrado perfecto
sqrtdt = np.sqrt(dt)
Npasos = int(Ttotal/dt)
ratio = 10**(1/Npasos)
Ntransiente = int(Transiente/dt)
Nceldas = int(L/radioc)
delta = L/Nceldas
coefphi = np.sqrt(2*D_r)
coefpos = np.sqrt(2*D_T)
# Funciones:

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
    moddif = np.array([dx, dy])
    rad = dx**2 + dy**2
    if rad < radioc2:
        fuerza = 48*epsilon*((2*sigma**12)/(rad**7))*moddif#48*epsilon*((2*sigma**12)/(rad**7))*moddif
        return fuerza
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
    global listaposiciones, scatter
    tiempo = dt*i
    data = listaposiciones[i]
    scatter.set_offsets(data)
    tiempo_text.set_text("t = %.2f" % tiempo)
    return scatter, tiempo_text


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
                        accel[i] += fuerzapar(particle, otherparticle)/masa
    phi += np.random.normal(N)*coefphi*sqrtdt
    if t % 100 == 0:
        termostato()
    
    velocidadinteractiva = velocidadinteractiva + accel*dt
    newarray = particles + (velocidadinteractiva + rotationarray(phi))*dt + np.random.normal(size=(N, 2))*coefpos*sqrtdt# Truquito pq o sino mod actuaba raro.
    particles = np.mod(newarray, L) # Para mantener las condiciones de borde.
    particles *= ratio
    L *= ratio
    Nceldas = int(L/radioc)
    delta = L/Nceldas
    if t >= Ntransiente:
        if t%frameskip == 0:
            listaposiciones.append(particles.copy())
            listavelocidades.append(velocidadinteractiva.copy())

listaposiciones = np.array(listaposiciones)
listavelocidades = np.array(listavelocidades)

fig = plt.figure(figsize=(7,7))
fig.clf()
ax = plt.axes(xlim=(0,L),ylim=(0,L))
scatter = ax.scatter(listaposiciones[0,0], listaposiciones[0,1])
tiempo_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
anim = FuncAnimation(fig, animable, frames=int(Npasos/frameskip), interval=33)
plt.show()
