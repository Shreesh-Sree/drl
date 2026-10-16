import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Regression
X = np.array([[0], [1], [2], [3]])
y = np.array([0, 1, 2, 3])

reg = Sequential([
    Dense(4, activation='relu', input_shape=(1,)),
    Dense(1)
])

reg.compile(optimizer='adam', loss='mse')
reg.fit(X, y, epochs=100, verbose=0)

print("Regression Output:")
print(reg.predict(X, verbose=0).round(2))

# Classification
X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

clf = Sequential([
    Dense(4, activation='relu', input_shape=(1,)),
    Dense(1, activation='sigmoid')
])

clf.compile(optimizer='adam', loss='binary_crossentropy')
clf.fit(X, y, epochs=200, verbose=0)

print("Classification Output:")
print((clf.predict(X, verbose=0) >= 0.5).astype(int))
