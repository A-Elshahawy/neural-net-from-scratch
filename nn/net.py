from .losses import Loss
from .layers import Layer, Function


class Net:

    __slots__ = ("layers", "loss")

    def __init__(self, layers, loss):
        assert isinstance(loss, Loss), "Loss must be an instance of nn.losses.Loss"
        for layer in layers:
            assert isinstance(layer, Function), "Layers must be an instance or nn.layers.Function or nn.layers.Layer"

        self.layers = layers
        self.loss = loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, x, y):
        return self.loss(x, y)

    def backward(self, x, y):
        d = self.loss.backward(x, y)
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def update_weights(self, learning_rate):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.weight_update(learning_rate)