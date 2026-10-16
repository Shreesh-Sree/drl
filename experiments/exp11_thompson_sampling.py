import numpy as np

p = [0.1, 0.7, 0.5]
s = np.zeros(3)
f = np.zeros(3)

for t in range(101):
    arm = np.argmax(np.random.beta(s + 1, f + 1))

    if np.random.rand() < p[arm]:
        s[arm] += 1
    else:
        f[arm] += 1

print("Successes:", s)
print("Failures:", f)
