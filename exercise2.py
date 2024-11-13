import numpy as np
import matplotlib.pyplot as plt

# Load Data
with open("psths.npz", "rb") as f:
    data = np.load(f)
    X, times = data["X"], data["times"]

# -- PART A --
def norm_rate(fireRates):
    '''
    Normalise the fireRates
    '''
    norm_rates = np.zeros_like(fireRates)
    a = fireRates.max(axis=(1, 2))
    b = fireRates.min(axis=(1, 2))
    for i, values in enumerate(a):
        norm_rates[i] = [(z - b[i]) / (values - b[i] + 5) 
                        for z in fireRates[i]]
    return norm_rates

# Plot Histogram of Max Rates
a = X.max(axis=(1, 2))
_, ax = plt.subplots()
ax.hist(a, bins=20, color=(0,102/255,204/255))

ax.set_xlabel("Max Firing Rate (Hz)")
ax.set_ylabel("Number of Neurons")
ax.set_title("Distribution of Max Firing Rates")
plt.show()

# -- PART B --
# Subtract Off Mean
def mean_center(norm_fire_rates):
    mu = np.mean(norm_fire_rates, axis=1)
    new_normal = np.moveaxis(norm_fire_rates, 1, 0)
    for i, values in enumerate(new_normal):
        new_normal[i] = values - mu
    norm_fire_rates = np.moveaxis(new_normal, 0, 1)
    return norm_fire_rates

normalised_FR = mean_center(norm_rate(X))

# Save Data
np.savez("psths_norm.npz", X=normalised_FR, times=times)

# -- PART C --
def PCA_Vm_Z(X):
    '''
    Perform PCA on input to return Vm and Z
    '''
    # Eigenvalues
    S = (1 / X.shape[1]) * X @ X.T
    #print("S shape:", S.shape)
    eigvalues, eigvectors = np.linalg.eig(S)
    eigvalues, eigvectors = eigvalues.real, eigvectors.real

    idx = eigvalues.argsort()[::-1]
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:, idx]
    eigvectors = eigvectors / np.linalg.norm(eigvectors, axis=0)

    # find V_m
    V_m = eigvectors[:, :12]
    #print("V_m:", V_m.shape)
    Z = V_m.T @ X
    Z = Z.reshape(12, 108, 46)
    
    return V_m, Z

data = np.load("psths_norm.npz")


time = data["times"]
mask = (time >= -150) & (time <= 300)
X = data["X"][:, :, mask]
X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
V_m, Z = PCA_Vm_Z(X)
np.savez("Exercise_2.npz", Z=Z, V=V_m, times=time[mask])