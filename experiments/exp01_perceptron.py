import numpy as np

class Perceptron:
    def __init__(self, n):
        self.weights = np.zeros(n + 1)

    def predict(self, x):
        s = np.dot(x, self.weights[1:]) + self.weights[0]
        return 1 if s >= 0 else 0

    def train(self, X, y):
        for _ in range(100):
            for x, label in zip(X, y):
                p = self.predict(x)
                self.weights[1:] += 0.01 * (label - p) * x
                self.weights[0] += 0.01 * (label - p)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

model = Perceptron(2)
model.train(X, y)

print("Prediction:", model.predict(np.array([1, 1])))
