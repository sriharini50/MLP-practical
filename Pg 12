import pandas as pd
import numpy as np

# Step 1: Read CSV
data = pd.read_csv("data.csv")

print("Dataset:")
print(data)

# Step 2: Separate input and output
X = data[["StudyHours", "Attendance"]].values
y = data[["Result"]].values

# Step 3: Normalize input
X = X / np.max(X, axis=0)

# Step 4: Initialize weights
np.random.seed(1)

W1 = np.random.rand(2, 2)
W2 = np.random.rand(2, 1)

# Step 5: Initialize bias
b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))

# Step 6: Learning rate
learning_rate = 0.5

# Step 7: Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step 8: Training
for i in range(5000):

    # Forward propagation
    hidden = sigmoid(np.dot(X, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)

    # Calculate error
    error = y - output

    # Back propagation
    output_delta = error * output * (1 - output)

    hidden_error = np.dot(output_delta, W2.T)

    hidden_delta = hidden_error * hidden * (1 - hidden)

    # Update weights
    W2 += np.dot(hidden.T, output_delta) * learning_rate
    W1 += np.dot(X.T, hidden_delta) * learning_rate

    # Update bias
    b2 += np.sum(
        output_delta,
        axis=0,
        keepdims=True
    ) * learning_rate

    b1 += np.sum(
        hidden_delta,
        axis=0,
        keepdims=True
    ) * learning_rate

# Step 9: Display prediction
print("\nPredicted Output:")
print(np.round(output, 2))
