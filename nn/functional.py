import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def leaky_relu_prime(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def elu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_prime(x, alpha=0.01):
    return np.where(x > 0, 1, alpha * np.exp(x))


def swish(x):
    return x * sigmoid(x)


def swish_prime(x):
    return x * sigmoid(x) + sigmoid(x)