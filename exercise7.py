import numpy as np
from exercise2 import norm_rate, mean_center, PCA_Vm_Z
from exercise5 import Z_proj,plot_trajectory

data = np.load("psths.npz")
X = data["X"]
times = data["times"]

idx = np.where(times == -150)[0][0]
for i, values in enumerate(X):
    C = np.random.choice(108, (108 // 2,), replace=False)
    for j in C:
        X[i][j][idx:] = 2 * X[i][j][idx] - X[i][j][idx:]

x_norm = norm_rate(X)
X = mean_center(x_norm)

time = data["times"]
mask = (time >= -150) & (time <= 300)
time = time[mask]
X = X[:, :, mask]
X_sh = X.shape
X = X.reshape(X_sh[0], X_sh[1] * X_sh[2])
V_m, Z = PCA_Vm_Z(X)

Z_proj, P = Z_proj(Z, 0)
plot_trajectory(Z_proj, "Distorted Data: Plane of Fastest Rotation")