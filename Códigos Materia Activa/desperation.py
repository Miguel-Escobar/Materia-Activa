import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np



filename = "Códigos Materia Activa\\Termalizado y expandido\\Hertzian\\0.9initialpf400particles10.0velocity5time0.npz"

data = np.load(filename)
store_positions = data["positions"]
store_boxsize = data["boxsizes"]
parameters = data["parameters"]
sqrtN = parameters[3]
gammaexpansion = parameters[8]
dt = parameters[-1]
ratio = (1 + gammaexpansion*dt)
frameskip = parameters[-2]
tipo = parameters[-3]
Nsteps = len(store_boxsize)
Nparticles = len(store_positions[0, :, 0])

def animable(i, scatter, ax, store_positions, store_boxsize, tiempo_text, marksize):
    global ind
    if gammaexpansion != 0:
        ax.set_xlim((0, store_boxsize[i]))
        ax.set_ylim((0, store_boxsize[i]))
        marksize = marksize/(ratio**(2*i*frameskip))
        scatter.set_sizes(marksize*np.ones(Nparticles))
    tiempo = dt*i*frameskip
    data = store_positions[i]
    scatter.set_offsets(data)
    tiempo_text.set_text("t = %.2f" % tiempo)
    if i % int(Nsteps/3) == 0 and ind <5:
        figanimable.savefig("expsnapshot%i.svg" % ind, transparent=True)
        ind += 1 

    return scatter, tiempo_text, ax
marksize = int(300/sqrtN)*int(300/sqrtN)/4.7
figanimable = plt.figure(figsize=(6.5,6.5))
figanimable.clf()
ax1 = figanimable.add_subplot(221, xlim=(0,store_boxsize[0]),ylim=(0,store_boxsize[0]))
ax1.set_xlabel("x [$\sigma$]")
ax1.set_ylabel("y [$\sigma$]")
scatter = ax1.scatter(store_positions[0,:,0], store_positions[0,:,1], s=marksize, c="firebrick")
tiempo = 0
tiempo_text = ax1.text(0.05, 0.9114, "t = %.2f" % tiempo, transform=ax1.transAxes, backgroundcolor='0.9')

i = int(Nsteps/3)
ax2 = figanimable.add_subplot(222, xlim=(0,store_boxsize[i]),ylim=(0,store_boxsize[i]))
marksize = marksize/(ratio**(2*i*frameskip))
tiempo = dt*i*frameskip
scatter = ax2.scatter(store_positions[i,:,0], store_positions[i,:,1], s=marksize, c="firebrick")
ax2.set_xlabel("x [$\sigma$]")
ax2.set_ylabel("y [$\sigma$]")
tiempo_text = ax2.text(1.3, 0.9114, "t = %.2f" % tiempo, transform=ax1.transAxes, backgroundcolor='0.9')

i = 2*int(Nsteps/3)
ax3 = figanimable.add_subplot(223, xlim=(0,store_boxsize[i]),ylim=(0,store_boxsize[i]))
marksize = marksize/(ratio**(2*i*frameskip))
tiempo = dt*i*frameskip
scatter = ax3.scatter(store_positions[i,:,0], store_positions[i,:,1], s=marksize, c="firebrick")
ax3.set_xlabel("x [$\sigma$]")
ax3.set_ylabel("y [$\sigma$]")
tiempo_text = ax3.text(0.05, -0.3, "t = %.2f" % tiempo, transform=ax1.transAxes, backgroundcolor='0.9')

i = 3*int(Nsteps/3)
ax4 = figanimable.add_subplot(224, xlim=(0,store_boxsize[i]),ylim=(0,store_boxsize[i]))
marksize = marksize/(ratio**(2*i*frameskip))
tiempo = dt*i*frameskip
scatter = ax4.scatter(store_positions[i,:,0], store_positions[i,:,1], s=marksize, c="firebrick")
ax4.set_xlabel("x [$\sigma$]")
ax4.set_ylabel("y [$\sigma$]")
tiempo_text = ax4.text(1.3, -0.3, "t = %.2f" % tiempo, transform=ax1.transAxes, backgroundcolor='0.9')
figanimable.tight_layout()
figanimable.savefig("sequence.svg", transparent=True)
