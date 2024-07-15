import numpy as np
from math import sqrt
from itertools import product
from .utils import zero_pad


class Function:
    """
    Abstract model of a differentiable function.
    """

    def __init__(self, *args, **kwargs):
        # initializing cache for intermediate results
        # helps with gradient calculation in some cases
        self.cache = {}
        # cache for gradients
        self.grad_cache = {}

    def __call__(self, *args, **kwargs):
        # calculating output
        output = self.forward(*args, **kwargs)
        # calculating and caching local gradients
        self.grad_cache = self.local_grad(*args, **kwargs)
        return output

    def forward(self, *args, **kwargs):
        """
        Forward pass of the function. Calculates the output value and the
        gradient at the input as well.
        """
        ...

    def backward(self, *args, **kwargs):
        """
        Backward pass. Computes the local gradient at the input value
        after forward pass.
        """
        ...

    def local_grad(self, *args, **kwargs):
        """
        Calculates the local gradients of the function at the given input.

        Returns:
            grad_cache: dictionary of local gradients.
        """
        ...


class Layer(Function):
    """
    Abstract model of a neural network layer. In addition to Function, a Layer
    also has weights and gradients with respect to the weights.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = {}
        self.weight_update = {}

    def _init_weights(self, *args, **kwargs):
        ...

    def _update_weights(self, lr):
        """
        Updates the weights using the corresponding _global_ gradients computed during
        backpropagation.

        Args:
            lr: float. Learning rate.
        """
        for key, weight in self.weight.items():
            self.weight[key] = self.weight[key] - lr * self.weight_update[key]


class Flatten(Function):
    def forward(self, X):
        self.cache["shape"] = X.shape
        n_batch = X.shape[0]
        return X.reshape(n_batch, -1)

    def backward(self, dY):
        return dY.reshape(self.cache["shape"])


class MaxPool2D(Function):
    def __init__(self, kernel_size=(2, 2)):
        super().__init__()
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )

    def __call__(self, X):
        # in contrary to other Function subclasses, MaxPool2D does not need to call
        # .local_grad() after forward pass because the gradient is calculated during it
        return self.forward(X)

    def forward(self, X):
        N, C, H, W = X.shape
        KH, KW = self.kernel_size

        grad = np.zeros_like(X)
        Y = np.zeros((N, C, H // KH, W // KW))

        # for n in range(N):
        for h, w in product(range(0, H // KH), range(0, W // KW)):
            h_offset, w_offset = h * KH, w * KW
            rec_field = X[:, :, h_offset: h_offset + KH, w_offset: w_offset + KW]
            Y[:, :, h, w] = np.max(rec_field, axis=(2, 3))
            for kh, kw in product(range(KH), range(KW)):
                grad[:, :, h_offset + kh, w_offset + kw] = (
                    X[:, :, h_offset + kh, w_offset + kw] >= Y[:, :, h, w]
                )

        # storing the gradient
        self.grad_cache["X"] = grad

        return Y

    def backward(self, dY):
        dY = np.repeat(
            np.repeat(dY, repeats=self.kernel_size[0], axis=2),
            repeats=self.kernel_size[1], axis=3
        )
        return self.grad_cache["X"] * dY

    def local_grad(self, X):
        # small hack: because for MaxPool calculating the gradient is simpler during
        # the forward pass, it is calculated there and this function just returns the
        # grad_cache dictionary
        return self.grad_cache


class BatchNorm2D(Layer):
    def __init__(self, in_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.in_channels = in_channels
        self._init_weights(in_channels)

    def _init_weights(self, in_channels):
        self.weight["gamma"] = np.ones(shape=(1, in_channels, 1, 1))
        self.weight["beta"] = np.zeros(shape=(1, in_channels, 1, 1))

    def forward(self, x):
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True) + self.eps
        invVar = 1.0 / var
        sqrt_invVar = np.sqrt(invVar)
        centered = x - mean
        scaled = centered * sqrt_invVar
        normalized = self.weight["gamma"] * scaled + self.weight["beta"]

        self.cache["mean"] = mean
        self.cache["var"] = var
        self.cache["invVar"] = invVar
        self.cache["sqrt_invVar"] = sqrt_invVar
        self.cache["centered"] = centered
        self.cache["scaled"] = scaled
        self.cache["normalized"] = normalized

        return normalized

    def backward(self, dY):
        d_gamma = np.sum(self.cache['scaled'] * dY, axis=(0, 2, 3), keepdims=True)
        d_beta = np.sum(dY, axis=(0, 2, 3), keepdims=True)

        self.weight_update['gamma'] = d_gamma
        self.weight_update['beta'] = d_beta
        dX = self.grad_cache['X'] * dY

        return dX

    def local_grad(self, X):
        N, C, H, W = X.shape

        ppc = H * W

        d_sqrt_invVar = self.cache['centered']
        d_invVar = (1.0 / (2.0 * np.sqrt(self.cache['invVar']))) * d_sqrt_invVar
        dd_var = (-1.0 / self.cache['var'] ** 2) * d_invVar
        d_denominator = (X - self.cache['mean']) * (2 * (ppc - 1) / ppc ** 2) * dd_var

        d_centered = self.cache['sqrt_invVar']
        d_numerator = (1.0 - 1.0 / ppc) * d_centered

        dX = d_numerator + d_denominator
        grads = {"X": dX}

        return grads


class Linear(Layer):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self._init_weights(in_dims, out_dims)

    def _init_weights(self, in_dims, out_dims):
        scale = 1 / sqrt(in_dims)
        self.weight["W"] = scale * np.random.randn(out_dims, in_dims)
        self.weight["b"] = scale * np.random.randn(1, out_dims)

    def forward(self, X):
        output = np.dot(X, self.weight["W"].T) + self.weight["b"]

        self.cache["X"] = X
        self.cache["output"] = output

        return output

    def backward(self, dY):
        dX = dY.dot(self.grad_cache["X"].T)
        X = self.cache["X"]
        dW = self.grad_cache["W"].T.dot(dY)
        db = np.sum(dY, axis=0, keepdims=True)

        self.weight_update = {"W": dW, "b": db}

        return dX

    def local_grad(self, X):
        gradX_local = self.weight["W"]
        gradW_local = X
        grad_b_local = np.ones_like(self.weight["b"])

        grads = {"X": gradX_local, "W": gradW_local, "b": grad_b_local}

        return grads


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding

        self._init_weights(in_channels, out_channels, kernel_size)

    def _init_weights(self, in_channels, out_channels, kernel_size):
        scale = 2 / sqrt(in_channels * kernel_size[0] * kernel_size[1])

        self.weight["W"] = scale * np.random.normal(scale=scale, size=(out_channels, in_channels, *kernel_size))
        self.weight["b"] = scale * np.random.randn(1, out_channels)

    def forward(self, X):
        if self.padding:
            X = zero_pad(X, self.padding, dims=(2, 3))

        self.cache["X"] = X

        N, C, H, W = X.shape
        KH, KW = self.kernel_size

        out_shape = (N, self.out_channels, 1 + (H - KH)//self.stride, 1 + (W - KW) // self.stride,
                     1 + (W - KW) // self.stride)

        Y = np.zeros(out_shape)

        for n in range(N):
            for c_w in range(self.out_channels):
                for h, w in product(range(out_shape[2]), range(out_shape[3])):
                    h_offset, w_offset = self.stride * h, self.stride * w
                    rec_field = X[n, :, h_offset:h_offset + KH, w_offset:w_offset + KW]
                    Y[n, c_w, h, w] = (
                        np.sum(self.weight["W"][c_w] * rec_field) + self.weight["b"][c_w]
                    )

        return Y

    def backward(self, dY):
        X = self.cache["X"]
        N, C, H, W = X.shape
        KH, KW = self.kernel_size

        dX = np.zeros(X.shape)
        # * this actually a Conv2D transpose
        for n in range(N):
            for c_w in range(self.out_channels):
                for h, w in product(range(dY.shape[2]), range(dY.shape[3])):
                    h_offset, w_offset = self.stride * h, self.stride * w
                    dX[n, :, h_offset:h_offset + KH, w_offset:w_offset + KW] += (
                        dY[n, c_w, h, w] * self.weight["W"][c_w]
                    )

        dw = np.zeros_like(self.weight["W"])
        for c_w in range(self.out_channels):
            for c_i in range(self.in_channels):
                for h, w in product(range(KH), range(KW)):
                    X_rec_field = X[:, c_i, h: H - KH + h + 1: self.stride, w: W - KW + w + 1: self.stride]
                    dY_rec_field = dY[:, c_w]

                    dw[c_w, c_i, h, w] = np.sum(X_rec_field * dY_rec_field)

        db = np.sum(dY, axis=(0, 2, 3)).reshape(-1, 1)

        self.weight_update = {"W": dw, "b": db}

        return dX[:, :, self.padding:-self.padding, self.padding:-self.padding]
