import numpy as np

n = 10
means = np.random.rand(n)
counts = np.zeros(n)
values = np.zeros(n)
total = 0

for t in range(1000):
    if 0 in counts:
        arm = np.argmin(counts)
    else:
        ucb = values + 2 * np.sqrt(np.log(t + 1) / counts)
        arm = np.argmax(ucb)

    reward = np.random.randn() + means[arm]
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    total += reward

print("Total reward:", round(total, 2))
