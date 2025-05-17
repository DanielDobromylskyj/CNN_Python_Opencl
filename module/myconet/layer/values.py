import numpy as np

from ..activations import *


def create_network_buffer(shape, activation_type, loading=False, testing=None):
    if testing is not None:
        return np.full(shape, testing[0], dtype=np.float32)

    elif loading:
        return None

    elif activation_type == ReLU:
        return create_relu_network_buffer(shape)

    elif activation_type == Sigmoid:
        return create_sigmoid_network_buffer(shape)

    else:
        raise NotImplementedError("Activation type not implemented in weight / bias creation!")


def mul(sizes):
    """ Helper Function """
    total = 1
    for size in sizes:
        total *= size

    return total


def create_relu_network_buffer(shape: tuple):
    if shape[0] <= 0:
        raise ValueError("Fan In cannot be less than or equal to 0.")

    random_values = np.random.randn(mul(shape))
    random_values = np.clip(random_values, -10, 10)
    return random_values.astype(np.float32) * np.sqrt(2.0 / shape[0])


def sigmoid_weight_init(shape: tuple):
    if shape[0] <= 0:
        raise ValueError("Fan In cannot be less than or equal to 0.")

    random_values = np.random.randn(np.prod(shape))
    random_values = np.clip(random_values, -10, 10)
    return random_values.astype(np.float32) * np.sqrt(2.0 / shape[0])
