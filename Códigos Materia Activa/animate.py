import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
#rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\Desktop\\ffmpeg\\bin\\ffmpeg.exe"
rcParams['animation.ffmpeg_path'] = "C:\\Users\\migue\\OneDrive\\Escritorio\\ffmpeg-2023-01-01-git-62da0b4a74-essentials_build\\bin\\ffmpeg.exe"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=2800)


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

figanimable = plt.figure(figsize=(6.5,6.5))
figanimable.clf()
ax = figanimable.add_subplot(xlim=(0,store_boxsize[0]),ylim=(0,store_boxsize[0]))
marksize = int(300/sqrtN)*int(300/sqrtN)
if gammaexpansion <= 0:
    marksize = marksize/((1 - gammaexpansion*dt)**(2*Nsteps*frameskip))
ax.set_xlabel("x [$\sigma$]")
ax.set_ylabel("y [$\sigma$]")
scatter = ax.scatter(store_positions[0,:,0], store_positions[0,:,1], s=marksize, c="firebrick")
tiempo_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
ind=0
anim = FuncAnimation(figanimable, animable, fargs=(scatter, ax, store_positions, store_boxsize, tiempo_text, marksize) ,frames=Nsteps, interval=10)
if input("Save animation? (y/n) ") == "y":
    if tipo == 1:
        anim.save('lennardjones%i.mp4' % Nparticles, writer=writer)
    if tipo == 2:
        anim.save("expansion.mp4", writer=writer)#'expansionharmonic%i.mp4' % Nparticles, writer=writer)
