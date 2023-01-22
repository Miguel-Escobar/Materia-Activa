import numpy as np
import matplotlib.pyplot as plt


dataexpanded = np.load("No termalizado si expandido\\Hertzian\\analyzed_data5.0velocity.npz")
datacontracted = np.load("No termalizado si contraido\\Hertzian\\analyzed_data5.0velocity.npz")
fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.errorbar(dataexpanded["packing_fractions"], dataexpanded["clusteringavg"], dataexpanded["clusteringstderr"], label="Expansión")
ax1.errorbar(datacontracted["packing_fractions"], datacontracted["clusteringavg"], datacontracted["clusteringstderr"], label="Contracción")
ax2.errorbar(dataexpanded["packing_fractions"], dataexpanded["maxclusteravg"], dataexpanded["maxclusterstderr"], label="Expansión")
ax2.errorbar(datacontracted["packing_fractions"], datacontracted["maxclusteravg"], datacontracted["maxclusterstderr"], label="Contracción")
ax1.set_xlabel("$\phi$")
ax2.set_xlabel("$\phi$")
ax1.set_ylabel(r"$\langle c_n(t) \rangle$")
ax2.set_ylabel(r"Largest Cluster Size")
ax1.legend()
ax2.legend()
fig.tight_layout()
fig.show()

