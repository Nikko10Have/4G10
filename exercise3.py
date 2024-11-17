import matplotlib.pyplot as plt
import numpy as np
import cond_color as cc

data = np.load("Exercise_2.npz")
Z = data["Z"]

colors = cc.get_colors(Z[0, :, 0], Z[1, :, 0])
_, ax = plt.subplots()

for i in range(108):
    ax.plot(Z[0, i, :], Z[1, i, :], color=colors[i], alpha=0.5)
    cc.plot_start(Z[0, i, 0], Z[1, i, 0], colors[i], ax=ax, markersize=20)
    cc.plot_end(Z[0, i, -1], Z[1, i, -1], colors[i], ax=ax, markersize=10)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Trajectories in the PC1-PC2 plane")
plt.show()