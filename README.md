# Neural Network Library

This is a lightweight neural network library implemented in Python, designed for educational purposes and small-scale machine learning projects. It provides a flexible framework for building and training various types of neural networks.

## Features



Modular architecture with separate modules for layers, activation functions, loss functions, and the main network

Support for various layer types including Linear, Convolutional (Conv2D), MaxPooling, BatchNormalization, and Flatten

Multiple activation functions: Sigmoid, ReLU, Leaky ReLU, Tanh, ELU, Swish, and Mish

Loss functions: Mean Squared Error (MSE) and Cross-Entropy

Automatic differentiation for backpropagation

Flexible network construction using a list of layers


## Installation

To use this library, simply clone the repository and import the necessary modules:


```
git clone https://github.com/your-username/neural-network-library.git
cd neural-network-library
```

## Usage

Here's a basic example of how to create and use a neural network:

```

from nn.layers import Linear, Conv2D, MaxPool2D, Flatten

from nn.activations import Relu, Softmax

from nn.losses import CrossEntropyLoss

from nn.net import Net


# Define your network architecture

layers = [

    Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),

    Relu(),

    MaxPool2D(kernel_size=2),

    Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),

    Relu(),

    MaxPool2D(kernel_size=2),

    Flatten(),

    Linear(in_dims=64*7*7, out_dims=128),

    Relu(),

    Linear(in_dims=128, out_dims=10),

    Softmax()

]


# Create the network

net = Net(layers, loss=CrossEntropyLoss())


# Forward pass

output = net.forward(input_data)


# Compute loss

loss = net.loss(output, target)


# Backward pass

net.backward(output, target)


# Update weights

net.update_weights(learning_rate=0.01)

```



## Module Overview

### [layers.py](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/layers.py)



Contains base classes for layers ([Function](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/layers.py#L7) and [Layer](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/layers.py#L50)) and implementations of specific layer types:


[Flatten](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/layers.py#L76)

[MaxPool2D](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/layers.py#L86)

[BatchNorm2D](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/layers.py#L134)

[Linear](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/layers.py#L193)

[Conv2D](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/layers.py#L233)


### [activations.py](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/532b7448b496a05696dfa6ad357b29c9cb23f269/nn/nn/activations.py)

Implements various activation functions as subclasses of Function:


[Sigmoid](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/532b7448b496a05696dfa6ad357b29c9cb23f269/nn/nn/activations.py#L6)

[Relu](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/532b7448b496a05696dfa6ad357b29c9cb23f269/nn/nn/activations.py#L18)

[LeakyRelu](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/532b7448b496a05696dfa6ad357b29c9cb23f269/nn/nn/activations.py#:30)

[Softmax](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/532b7448b496a05696dfa6ad357b29c9cb23f269/nn/nn/activations.py#L42)

[Tanh](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/532b7448b496a05696dfa6ad357b29c9cb23f269/nn/nn/activations.py#L73)

[Elu](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/532b7448b496a05696dfa6ad357b29c9cb23f269/nn/nn/activations.py#L85)

[Swish](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/532b7448b496a05696dfa6ad357b29c9cb23f269/nn/nn/activations.py#L101)

[Mish](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/532b7448b496a05696dfa6ad357b29c9cb23f269/nn/nn/activations.py#L113)


### [functional.py](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/85a84991b1d20c63f32b1f5f9c9e7c997297595d/nn/nn/functional.py)

Provides numpy implementations of activation functions and their derivatives.

### [losses.py](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/losses.py)

Implements loss functions:


[MSELoss](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/losses.py#L46) (Mean Squared Error)

[CrossEntropyLoss](https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/8c3aeff96e1008d6e7ad61f9950e5cb9753ef22e/nn/nn/losses.py#L57)


### [net.py]([https://github.com/A-Elshahawy/neural-nets-from-scratch/blob/cec13eb7d1824205490e48d4437a62fd5c946e1d/nn/nn/net.py](https://github.com/A-Elshahawy/neural-net-from-scratch/blob/72d6b9ecadcf928315905f3cf89baceae27d4cad/nn/net.py))

Defines the Net class for creating and managing the entire neural network.

## _Contributing_

Contributions to this project are welcome! Please feel free to submit a Pull Request.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This library is primarily for educational purposes and may not be optimized for large-scale or production use. For more robust deep learning frameworks, consider using PyTorch, TensorFlow, or other established libraries.

