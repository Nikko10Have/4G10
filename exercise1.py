import numpy as np
import matplotlib.pyplot as plt

with open("psths.npz", "rb") as f:
    data = np.load(f)
    X, times = data["X"], data["times"]

fig, ax = plt.subplots()
colors = (np.array([
            [255, 0, 0],
            [220, 220, 0],
            [0, 128, 155],
            [0, 255, 0]
            ])/ 255
        )
line_styles = ["-", "--", "-.", ":"]

z = np.random.choice(100, 4, replace=0)
q = np.random.choice(180, 4, replace=0)

#Plot PSTHS 
for a, neuron in enumerate(q):
    for b, condition in enumerate(z):
        ax.plot(times, X[neuron][condition],
                color=colors[a], linestyle=line_styles[b])

#Setup legend
hand = []
lab = []
for i, neuron in enumerate(q):
    hand.append(plt.Line2D([0], [0], color=colors[i], lw=2))
    lab.append(f"Neuron {neuron}")
for j, condition in enumerate(z):
    hand.append(plt.Line2D([0], [0], color="black", lw=2, linestyle=line_styles[j]))
    lab.append(f"Condition {condition}")

# Legend and Labels
ax.legend(hand, lab, ncol=4, fontsize="x-small", markerscale=0.5)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Firing Rate (Hz)")
ax.set_title("PSTHs")
plt.show()

# Population Average
pop_avg = np.mean(X, axis=(0, 1))

fig, ax = plt.subplots()
ax.plot(times, pop_avg, color = colors[3])
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Average Firing rate (Hz)")
ax.set_title("Average Firing Rate over all Neurons & Conditions")
plt.show()