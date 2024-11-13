import numpy as np
from exercise2 import norm_rate, mean_center, PCA_Vm_Z
from exercise5 import Z_proj,plot_trajectory

data = np.load("psths.npz")
X = data["X"]
times = data["times"]

idx = np.where(times == -150)[0][0]
print(idx)
print(f"time: {times[idx]}")

for i, values in enumerate(X):
    C = np.random.choice(108, (108 // 2,), replace=False)
    for j in C:
        X[i][j][idx:] = 2 * X[i][j][idx] - X[i][j][idx:]

x_normal = norm_rate(X)
X = mean_center(x_normal)

time = data["times"]
mask = (time >= -150) & (time <= 300)
time = time[mask]
X = X[:, :, mask]
X_shape = X.shape
X = X.reshape(X_shape[0], X_shape[1] * X_shape[2])
V_m, Z = PCA_Vm_Z(X)

Z_proj, P = Z_proj(Z, 0)
plot_trajectory(Z_proj, "Plane of 1st FR Distorted Data")