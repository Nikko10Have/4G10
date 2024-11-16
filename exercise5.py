from exercise4 import A_mat
import numpy as np
import matplotlib.pyplot as plt
import cond_color

def Z_proj(Z, i):
    A = A_mat(Z)

    # Get eigenvalues and vectors
    eigvalues, eigvectors = np.linalg.eig(A)
    eigvectors = eigvectors[:,::2]
    eigvalues = eigvalues[::2]
    idx = eigvalues.imag.argsort()[::-1]
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:, idx]

    for n in range(len(eigvalues)):
        print(eigvalues[n], ": \n ", eigvectors[n], "\n")
    # Find P with largest eigenval
    P = (np.stack(((eigvectors[:, i].real) / np.linalg.norm(eigvectors[:, i].real),
            eigvectors[:, i].imag / np.linalg.norm(eigvectors[:, i].imag),),axis=1)).T

    Zsh = Z.shape
    Z = Z.reshape(Zsh[0], Zsh[1] * Zsh[2])
    Z_pr = P @ Z
    Z_pr = Z_pr.reshape(2, Zsh[1], Zsh[2])

    return Z_pr, P

def plot_trajectory(Z_proj, title, alt=False):
    colors = cond_color.get_colors(Z_proj[0, :, 0], Z_proj[1, :, 0], alt_colors=alt)

    # Plot the trajectories for all conditions in the same plot
    _, ax = plt.subplots()
    for i in range(108):
        ax.plot(Z_proj[0, i, :], Z_proj[1, i, :], color=colors[i])
        cond_color.plot_start(Z_proj[0, i, 0], Z_proj[1, i, 0],
            colors[i], ax=ax, markersize=20)
        cond_color.plot_end(Z_proj[0, i, -1], Z_proj[1, i, -1],
            colors[i], ax=ax, markersize=10)
    ax.set_xlabel("real")
    ax.set_ylabel("imag")
    ax.set_title(title)
    plt.show()

data = np.load("Exercise_2.npz")
Z = data["Z"]
time = data["times"]

mask = (time >= -150) & (time <= 200)
Z = Z[:, :, mask]


Z_1, P_fr = Z_proj(Z, 0)
plot_trajectory(Z_1, "Fastest Rotation")
#Z_2, P_fr1 = Z_proj(Z, 1)
#plot_trajectory(Z_2, "2nd Fastest Rotation ")
#Z_3, P_fr2 = Z_proj(Z, 2)
#plot_trajectory(Z_3, "3rd Fastest Rotation ")

#np.savez("Exercise_5.npz", P=[P_fr, P_fr1, P_fr2], Z_1=Z_1, Z_2=Z_2, Z_3=Z_3)