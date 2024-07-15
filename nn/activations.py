import numpy as np
from .functional import *
from .layers import Function


class Sigmoid(Function):
    def forward(self, X):
        return sigmoid(X)

    def backward(self, dY):
        return dY * self.grad_cache['X']

    def local_grad(self, X):
        grads = {'X': sigmoid_prime(X)}
        return grads


class Relu(Function):
    def forward(self, X):
        return relu(X)

    def backward(self, dY):
        return dY * self.grad_cache['X']

    def local_grad(self, X):
        grads = {'X': relu_prime(X)}
        return grads


class LeakyRelu(Function):
    def forward(self, X):
        return leaky_relu(X)

    def backward(self, dY):
        return dY * self.grad_cache['X']

    def local_grad(self, X):
        grads = {'X': leaky_relu_prime(X)}
        return grads


class Softmax(Function):
    def forward(self, X):
        exp_x = np.exp(X)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.cache['X'] = X
        self.cache['output'] = probs
        return probs

    def backward(self, dY):
        dX = []

        for dy_row, grad_row in zip(dY, self.grad_cache['X']):
            dX_row = np.dot(dy_row, grad_row)
            dX.append(dX_row)

        return np.array(dX)

    def local_grad(self, X):
        grads = []

        for prob in self.cache['output']:
            prob = prob.reshape(-1, 1)
            grad_row = - np.dot(prob, prob.T)
            grad_row_diagonal = - prob * (1 - prob)
            np.fill_diagonal(grad_row, grad_row_diagonal)
            grads.append(grad_row)

        grad = np.array(grads)
        return {'X': grad}


class Tanh(Function):
    def forward(self, X):
        return tanh(X)

    def backward(self, dY):
        return dY * self.grad_cache['X']

    def local_grad(self, X):
        grads = {'X': tanh_prime(X)}
        return grads


class Elu(Function):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, X):
        return elu(X, self.alpha)

    def backward(self, dY):
        return dY * self.grad_cache['X']

    def local_grad(self, X):
        grads = {'X': elu_prime(X, self.alpha)}
        return grads


class Swish(Function):
    def forward(self, X):
        return swish(X)

    def backward(self, dY):
        return dY * self.grad_cache['X']

    def local_grad(self, X):
        grads = {'X': swish_prime(X)}
        return grads


class Mish(Function):
    def forward(self, X):
        return X * np.tanh(np.log(1 + np.exp(X)))

    def backward(self, dY):
        return dY * self.grad_cache['X']

    def local_grad(self, X):
        exp_x = np.exp(X)
        exp_2x = np.exp(2 * X)
        exp_3x = np.exp(3 * X)
        omega = 4 * (X + 1) + 4 * exp_2x + exp_3x + exp_x * (4 * X + 6)
        delta = 2 * exp_x + exp_2x + 2
        grads = {'X': (exp_x * omega) / (delta * delta)}
        return grads
