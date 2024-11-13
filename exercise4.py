import numpy as np
import matplotlib.pyplot as plt

def gen_H(M):
    K = int(M * (M - 1) / 2)
    H = np.zeros((K, M, M))
    a = 1
    i = 0
    while i < M - 1:
        while a <= (K - ((M - 1 - i) * (M - 2 - i) / 2)):
            a_0 = (K - ((M - i) * (M - i - 1) / 2)) + 1
            ind = int(a - a_0 + 1 + i)
            H[a - 1][i][ind] = 1
            H[a - 1][ind][i] = -1
            a += 1
        i += 1
    return H

def A_mat(Z):
    delta_Z = np.diff(Z)
    delta_Z = np.reshape(delta_Z, (delta_Z.shape[0], delta_Z.shape[1] * delta_Z.shape[2]))
    Z = np.reshape(Z[:, :, 0:-1], (Z.shape[0], Z.shape[1] * Z.shape[2] - Z.shape[1]))

    H = gen_H(Z.shape[0])
    W = np.tensordot(H, Z, axes=1)
    b = np.tensordot(W, delta_Z, axes=([1, 2], [0, 1]))
    Q = np.tensordot(W, W, axes=([1, 2], [1, 2]))

    Beta = np.linalg.solve(Q, b)

    A = np.tensordot(Beta, H, axes=1)
    return A

data = np.load("test.npz")
print(data)
Z_test = data["Z_test"]
A_test = data["A_test"]
A = A_mat(Z_test)
delta_A = A_test - A
print(np.max(delta_A))

fig, ax = plt.subplots()
im = ax.imshow(delta_A, cmap="BuPu_r")
ax.set_title("A")
fig.colorbar(im)
plt.show()
