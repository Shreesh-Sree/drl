# Deep RL Lab: Python Experiments

Aim, algorithm, program, output and result for each Python experiment. Code lives in [`experiments/`](experiments/), one file per experiment.

## Contents

| # | Experiment | Language | File |
|---|---|---|---|
| 1 | Implementation of Perceptron Algorithm | Python | [`exp01_perceptron.py`](experiments/exp01_perceptron.py) |
| 2 | Classification Using Backpropagation | Python | [`exp02_backpropagation_xor.py`](experiments/exp02_backpropagation_xor.py) |
| 4 | Deep Neural Network for Classification and Regression | Python | [`exp04_dnn_regression_classification.py`](experiments/exp04_dnn_regression_classification.py) |
| 5 | Image Resizing, Rotation, and Preprocessing | Python | [`exp05_image_preprocessing.py`](experiments/exp05_image_preprocessing.py) |
| 6 | Text Classification Using CNN | Python | [`exp06_cnn_text_classification.py`](experiments/exp06_cnn_text_classification.py) |
| 9 | Epsilon-Greedy Algorithm | Python | [`exp09_epsilon_greedy.py`](experiments/exp09_epsilon_greedy.py) |
| 10 | Upper Confidence Bound (UCB) Algorithm | Python | [`exp10_ucb.py`](experiments/exp10_ucb.py) |
| 11 | Thompson Sampling Algorithm | Python | [`exp11_thompson_sampling.py`](experiments/exp11_thompson_sampling.py) |

## Setup

- **Python experiments (1, 2, 4, 5, 6, 9, 10, 11):** Python 3.9+, then `pip install -r requirements.txt`.
- **Numbering:** experiments 3, 7 and 8 (MATLAB) are not included, so the numbers below skip them.
- **Experiment 5:** place `opera_house.jpg` in the same folder as the script.
- Run a Python experiment with `python experiments/exp01_perceptron.py`.

## Experiment 1: Implementation of Perceptron Algorithm

**Aim:** To implement the Perceptron algorithm using Python.

**Algorithm**

1. Initialize weights and bias to zero.
2. Calculate the weighted sum of inputs.
3. Apply the activation function.
4. Update weights and bias based on prediction error.
5. Train the model and predict the output.

**Program** (`experiments/exp01_perceptron.py`)

```python
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
```

**Output**

```
Prediction: 1
```

**Result:** The Perceptron algorithm was implemented successfully and predicted the output for the AND gate.

## Experiment 2: Classification Using Backpropagation

**Aim:** To implement classification using backpropagation.

**Algorithm**

1. Initialize the training inputs and target outputs.
2. Construct a neural network with input, hidden, and output layers.
3. Perform forward propagation.
4. Calculate the prediction error.
5. Update the network weights using backpropagation.
6. Predict and display the output.

**Program** (`experiments/exp02_backpropagation_xor.py`)

```python
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = Sequential([
    Dense(4, activation='sigmoid', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=1000, verbose=0)

print(np.round(model.predict(X, verbose=0)))
```

**Output**

```
[[0.]
 [1.]
 [1.]
 [0.]]
```

**Result:** The neural network was trained using backpropagation to learn the XOR classification problem.

> **Note:** Neural network predictions can vary because of random weight initialization.

## Experiment 4: Deep Neural Network for Classification and Regression

**Aim:** To implement deep neural networks for classification and regression using Python.

**Algorithm**

1. Import NumPy and the required Keras libraries.
2. Define the input and target data.
3. Construct a neural network with hidden and output layers.
4. Compile the model using an optimizer and loss function.
5. Train the model.
6. Predict and display the output.

**Program** (`experiments/exp04_dnn_regression_classification.py`)

```python
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
```

**Output**

```
Regression Output:
[[0.05]
 [0.98]
 [1.95]
 [2.98]]

Classification Output:
[[0]
 [0]
 [1]
 [1]]
```

**Result:** Deep neural networks were implemented for both regression, which predicts continuous numerical values, and binary classification, which predicts class labels 0 or 1.

> **Note:** Exact regression values may vary slightly between runs.

## Experiment 5: Image Resizing, Rotation, and Preprocessing

**Aim:** To perform basic image preprocessing operations using Python.

**Algorithm**

1. Import the required libraries.
2. Load an image.
3. Display the image dimensions.
4. Resize the image.
5. Flip the image horizontally.
6. Rotate the image by 45 degrees.
7. Display all processed images.

**Program** (`experiments/exp05_image_preprocessing.py`)

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = Image.open('opera_house.jpg')

# Print image dimensions and color channels
print(np.array(img).shape)

# Display the original image
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Original")

# Resize the image to 500 x 500 pixels
plt.subplot(2, 2, 2)
plt.imshow(img.resize((500, 500)))
plt.title("Resized")

# Flip the image horizontally
plt.subplot(2, 2, 3)
plt.imshow(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
plt.title("Flipped")

# Rotate the image by 45 degrees
plt.subplot(2, 2, 4)
plt.imshow(img.rotate(45))
plt.title("Rotated")

# Display all four images
plt.show()
```

**Output**

```
The program displays:
Original image
Resized image
Horizontally flipped image
Image rotated by 45 degrees
It also prints the original image dimensions, for example:
(600, 800, 3)
```

**Result:** Image resizing, flipping, rotation, and dimension inspection were performed successfully.

> **Note:** Requirement: keep `opera_house.jpg` in the same folder as the Python program.

## Experiment 6: Text Classification Using CNN

**Aim:** To classify text using a Convolutional Neural Network.

**Algorithm**

1. Load the IMDb sentiment dataset.
2. Pad the text sequences to a fixed length.
3. Construct a CNN using embedding and convolution layers.
4. Compile and train the model.
5. Evaluate the model on the test dataset.
6. Display the accuracy.

**Program** (`experiments/exp06_cnn_text_classification.py`)

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.utils import pad_sequences

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)

model = Sequential([
    Embedding(5000, 16),
    Conv1D(16, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

print("Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])
```

**Output**

```
Example:
Accuracy: 0.75
The accuracy varies depending on training conditions.
```

**Result:** A CNN model was trained to classify IMDb reviews as positive or negative.

> **Note:** Requirement: install TensorFlow using `pip install tensorflow`. The first run downloads the dataset.

## Experiment 9: Epsilon-Greedy Algorithm

**Aim:** To implement the Epsilon-Greedy algorithm using Python.

**Algorithm**

1. Initialize the slot machines and their estimated rewards.
2. Set the exploration probability, epsilon.
3. Select a random machine during exploration.
4. Otherwise, select the machine with the highest estimated reward.
5. Observe the reward and update the estimate.
6. Repeat for 1,000 trials.
7. Display the estimated rewards and best machine.

**Program** (`experiments/exp09_epsilon_greedy.py`)

```python
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
```

**Output**

```
Example:
Estimated rewards: [0.15 0.35 0.55 0.65 0.80]
Selection counts: [20. 30. 40. 50. 860.]
Best machine: 5
```

**Result:** The Epsilon-Greedy algorithm was implemented to balance exploration and exploitation while selecting the best-performing machine.

> **Note:** The displayed values are illustrative; actual output varies.

## Experiment 10: Upper Confidence Bound (UCB) Algorithm

**Aim:** To implement the Upper Confidence Bound algorithm using Python.

**Algorithm**

1. Initialize the arms and their estimated rewards.
2. Select each untried arm first.
3. Calculate the UCB score for each arm.
4. Select the arm with the highest UCB score.
5. Observe the reward and update the estimated value.
6. Repeat for 1,000 trials.
7. Display the total reward.

**Program** (`experiments/exp10_ucb.py`)

```python
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
```

**Output**

```
Example:
Total reward: 757.02
```

**Result:** The UCB algorithm was implemented to balance exploration and exploitation using confidence bounds.

> **Note:** The total reward varies between runs.

## Experiment 11: Thompson Sampling Algorithm

**Aim:** To implement the Thompson Sampling algorithm using Python.

**Algorithm**

1. Initialize the success and failure counts for each arm.
2. Sample a value from each arm's Beta distribution.
3. Select the arm with the highest sampled value.
4. Generate a reward based on the selected arm's probability.
5. Update the success or failure count.
6. Repeat for 101 trials.
7. Display the success and failure counts.

**Program** (`experiments/exp11_thompson_sampling.py`)

```python
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
```

**Output**

```
Example:
Successes: [ 2. 48.  8.]
Failures: [ 8. 22. 13.]
```

**Result:** Thompson Sampling was implemented using Beta distributions to select arms and update their success and failure counts.

> **Note:** The output varies because arm selection and rewards are random.

## Quick Revision Table

| Experiment | Topic | Main output |
|---|---|---|
| 1 | Perceptron | Predicted class |
| 2 | Backpropagation | XOR predictions |
| 4 | Deep neural network regression | Predicted numerical values |
| 5 | Image preprocessing | Original and transformed images |
| 6 | CNN text classification | Test accuracy |
| 9 | Epsilon-Greedy | Estimated rewards and best machine |
| 10 | UCB | Total reward |
| 11 | Thompson Sampling | Success and failure counts |
