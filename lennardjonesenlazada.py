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
sqrtN=5
N = sqrtN*sqrtN
L=int(np.sqrt(N*sigma**2)) # Tamaño de la caja. Puedo poner int pq N es un cuadrado perfecto
radioc=2.5*sigma 
radioc2=radioc**2
Temperatura=5
Ttotal = 10
Transiente = 5
dt= 0.01
Npasos = int(Ttotal/dt)
Ntransiente = int(Transiente/dt)
Nceldas = int(L/radioc)
delta = L/Nceldas
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
        fuerza = -48*epsilon*((sigma**6)/(rad**4) - (2*sigma**12)/(rad**7))*moddif
        return fuerza
    else:
        return 0

def condicioninicial():
    global particles, velocities
    velocities = np.sqrt(2*Temperatura/masa)*norm.rvs(size=(N,2))
    velocities[:, 0] = velocities[:,0] - np.mean(velocities[:,0])
    velocities[:, 1] = velocities[:,1] - np.mean(velocities[:,1])
    lin, espaciado = np.linspace(0, L, sqrtN, endpoint=False, retstep=True)
    particles[:, 0] = np.meshgrid(lin,lin)[0].flatten() + espaciado/2
    particles[:, 1] = np.meshgrid(lin,lin)[1].flatten() + espaciado/2
    return

def termostato():
    global velocities
    vx = velocities[:,0]
    vy = velocities[:,1]
    Tcinetica = np.mean((masa/2)*(vx**2+vy**2))
    ajuste = np.sqrt(Temperatura/Tcinetica)
    velocities = velocities*ajuste
    return


def animable(i):
    global listaposiciones, scatter
    Tcinetica = np.mean((masa/2)*np.sum(listavelocidades[i]**2, axis=1))
    ajuste = np.sqrt(Temperatura/Tcinetica)
    tiempo = dt*i
    data = listaposiciones[i]
    scatter.set_offsets(data)
    delta_text.set_text("delta = %.1f" % ajuste)
    tiempo_text.set_text("t = %.2f" % tiempo)
    return scatter, delta_text, tiempo_text

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
particles = np.zeros((N, 2))
velocities = np.zeros((N,2))
condicioninicial()
termostato()

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
    if t % 100 == 0:
        termostato()
    
    velocities = velocities + accel*dt
    newarray = particles + velocities*dt # Truquito pq o sino mod actuaba raro.
    particles = np.mod(newarray, L) # Para mantener las condiciones de borde.

    if t >= Ntransiente:
        listaposiciones.append(particles.copy())
        listavelocidades.append(velocities.copy())

listaposiciones = np.array(listaposiciones)
listavelocidades = np.array(listavelocidades)

fig = plt.figure(figsize=(7,7))
fig.clf()
ax = plt.axes(xlim=(0,L),ylim=(0,L))
scatter = ax.scatter(listaposiciones[0,0], listaposiciones[0,1])
delta_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
tiempo_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
anim = FuncAnimation(fig, animable, frames=Npasos, interval=33)
plt.show()
