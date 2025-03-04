import numpy as np
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

def backpropagation(X, y, weights1, weights2, learning_rate, epochs):

  for epoch in range(epochs):

    layer1_output = sigmoid(np.dot(X, weights1))
    layer2_output = sigmoid(np.dot(layer1_output, weights2))

    error = y - layer2_output

    layer2_delta = error * sigmoid_derivative(layer2_output)
    layer1_delta = layer2_delta.dot(weights2.T) * sigmoid_derivative(layer1_output)

    weights2 += layer1_output.T.dot(layer2_delta) * learning_rate
    weights1 += X.T.dot(layer1_delta) * learning_rate

  return weights1, weights2

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  

weights1 = np.random.rand(2, 4) 
weights2 = np.random.rand(4, 1)  

learning_rate = 0.1
epochs = 10000

trained_weights1, trained_weights2 = backpropagation(X, y, weights1, weights2, learning_rate, epochs)

layer1_output = sigmoid(np.dot(X, trained_weights1))
layer2_output = sigmoid(np.dot(layer1_output, trained_weights2))
print("Predictions:")
print(layer2_output)