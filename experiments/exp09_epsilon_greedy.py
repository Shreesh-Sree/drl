import numpy as np

rewards = [1, 2, 3, 4, 5]
counts = np.zeros(5)
values = np.zeros(5)
epsilon = 0.1

for _ in range(1000):
    if np.random.random() < epsilon:
        action = np.random.randint(5)
    else:
        action = np.argmax(values)

    reward = np.random.binomial(1, rewards[action] / 5)
    counts[action] += 1
    values[action] += (reward - values[action]) / counts[action]

print("Estimated rewards:", values)
print("Selection counts:", counts)
print("Best machine:", np.argmax(values) + 1)
