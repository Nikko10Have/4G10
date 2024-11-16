import numpy as np
import matplotlib.pyplot as plt
import cond_color

data_1 = np.load("Exercise_2.npz")
Z = data_1["Z"]
V_m = data_1["V"]

data_2 = np.load("Exercise_5.npz")
P = data_2["P"]
Z = data_2["Z_1"]

data_3 = np.load("psths_norm.npz")
X = data_3["X"]
times = data_3["times"]

mask = (times >= -800) & (times <= 300)
idx = np.where(times == -150)[0][0]
times = times[mask]
X = X[:, :, mask]
X_sh = X.shape

for q in range(3):
    Z_n = V_m.T @ X.reshape(X_sh[0], X_sh[1] * X_sh[2])
    Z_proj = P[q] @ Z_n
    Z_proj = Z_proj.reshape(2, X_sh[1], X_sh[2])

    colors_1 = cond_color.get_colors(Z_proj[0, :, idx],
                                     Z_proj[1, :, idx],
                                     alt_colors=1)
    _, ax = plt.subplots()

    for i in range(30):
        ax.plot(Z_proj[0, i, : idx + 1], Z_proj[1, i, : idx + 1],
                color=colors_1[i], alpha=0.5)
        cond_color.plot_start(Z_proj[0, i, 0], Z_proj[1, i, 0],
                              colors_1[i], ax=ax, markersize=15)
        cond_color.plot_end(Z_proj[0, i, idx], Z_proj[1, i, idx],
                            colors_1[i], ax=ax, markersize=10)

    colors_2 = cond_color.get_colors(Z_proj[0, :, idx],
                                     Z_proj[1, :, idx])
    for i in range(30):
        ax.plot(Z_proj[0, i, idx:], Z_proj[1, i, idx:],
                color=colors_2[i], alpha=0.2)
        cond_color.plot_start(Z[0, i, 0], Z[1, i, 0],
                              colors_2[i], ax=ax, markersize=15)
        cond_color.plot_end(Z_proj[0, i, -1], Z_proj[1, i, -1],
                            colors_2[i], ax=ax, markersize=10)

    ax.set_xlabel("real")
    ax.set_ylabel("imag")
    ax.set_title(f"Fastest Rotation: {q+1}")
    plt.show()