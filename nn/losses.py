import numpy as np
from .layers import Function


class Loss(Function):
    def forward(self, X, Y):
        """
                Computes the loss of x with respect to y.

                Args:
                    X: numpy.ndarray of shape (n_batch, n_dim).
                    Y: numpy.ndarray of shape (n_batch, n_dim).

                Returns:
                    loss: numpy.float.
                """
        ...

    def backward(self, X, Y):
        """
                Computes the gradient of the loss with respect to x.

                Args:
                    X: numpy.ndarray of shape (n_batch, n_dim).
                    Y: numpy.ndarray of shape (n_batch, n_dim).

                Returns:
                    dX: numpy.ndarray of shape (n_batch, n_dim).
                """
        return self.grad_cache['X']

    def local_grad(self, X, Y):
        """
        Calculates the local gradients of the function at the given input.

        Args:
            X: numpy.ndarray of shape (n_batch, n_dim).
            Y: numpy.ndarray of shape (n_batch, n_dim).

        Returns:
            grad_cache: dictionary of local gradients.
        """
        ...


class MSELoss(Loss):
    def forward(self, X, Y):
        sum = np.sum((X - Y) ** 2, axis=1, keepdims=True)
        mse_loss = np.mean(sum)
        return mse_loss
        # return np.mean(np.square(X - Y))

    def local_grad(self, X, Y):
        return {'X': 2 * (X - Y)/X.shape[0]}


class CrossEntropyLoss(Loss):
    def forward(self, X, Y):
        """
            Computes the cross entropy loss of x with respect to y.

            Args:
                X: numpy.ndarray of shape (n_batch, n_dim).
                y: numpy.ndarray of shape (n_batch, 1). Should contain class labels
                    for each data point in x.

            Returns:
                crossentropy_loss: numpy.float. Cross entropy loss of x with respect to y.
            """
        exp_x = np.exp(X)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        log_probs = - np.log([probs[i, Y[i]] for i in range(len(probs))])
        ce_loss = np.mean(log_probs)

        self.cache['Y'] = Y
        self.cache['probs'] = probs

        return ce_loss
        # return np.mean(-np.log(X) * Y)

    def local_grad(self, X, Y):
        probs = self.cache['probs']
        ones = np.zeros_like(probs)
        for row_idx, col_idx in enumerate(Y):
            ones[row_idx, col_idx] = 1.0

        grads = {'X': (probs - ones) / float(len(X))}
        return grads