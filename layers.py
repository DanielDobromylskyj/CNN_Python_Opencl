from typing import List, Any

import pyopencl as cl
import numpy as np
import math

from buffers import NetworkBuffer, Gradients

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


# Todo - Make Sigmoid activation weight distribution and implement it into making a full pop layer


def load_core(core_element):
    with open(f"./core/{core_element}.txt", "r") as f:
        return cl.Program(ctx, f.read()).build()


def mul(sizes):
    total = 1
    for size in sizes:
        total *= size

    return total


def relu_weight_init(size, fan_in, dtype=np.float32):
    return np.random.randn(mul(size)).astype(dtype=dtype) * np.sqrt(2.0 / fan_in)


class Layer:
    def forward(self, inputs: NetworkBuffer):
        """ Perform a forward pass over the layer """
        pass

    def backward(self, inputs: NetworkBuffer, outputs: NetworkBuffer, output_error_gradients: NetworkBuffer,
                 learning_rate: float) -> tuple[Gradients, Gradients, Gradients]:
        """ Calculate the gradients and errors for weights, biases and the 'next' layer """
        pass

    def apply_gradients(self, weight_gradients: Gradients, bias_gradients: Gradients) -> None:
        """ Applies the gradients we have calculated """
        pass

    def get_total_nodes_in(self):
        raise NotImplementedError("Layer does not have a implemented 'get_total_nodes_in' method")

    def get_total_nodes_out(self):
        raise NotImplementedError("Layer does not have a implemented 'get_total_nodes_out' method")


class ConvolutedLayer(Layer):
    def __init__(self, input_size: tuple[int, int], kernel_size: tuple[int, int], filter_count: int,
                 colour_depth: int = 1, loading: bool = False):
        self.__filter_count = filter_count
        self.__input_size = input_size
        self.__kernel_size = kernel_size
        self.__colour_depth = colour_depth

        self.forward_core = load_core("kernel")

        # Don't spend computing power generating weights to just overwrite them when loading.
        self.weights = [
            NetworkBuffer(relu_weight_init((self.get_weight_count(),), self.get_weight_count()), self.get_weight_count())
            for i in range(self.__filter_count)
        ] if loading is False else None

        self.biases = [
            NetworkBuffer(np.zeros((1,), dtype=np.float32), self.get_bias_count()) if loading is False else None
            for i in range(self.__filter_count)
        ] if loading is False else None

    def get_filter_count(self):
        return self.__filter_count

    def get_true_kernel_shape(self) -> tuple[int, int]:
        return self.__kernel_size[0] * self.__colour_depth, self.__kernel_size[1]

    def get_true_input_shape(self) -> tuple[int, int]:
        return self.__input_size[0] * self.__colour_depth, self.__input_size[1]

    def get_output_shape(self) -> tuple[int, int]:
        kernel_size = self.get_true_kernel_shape()
        input_size = self.get_true_input_shape()

        return math.ceil(input_size[0] / kernel_size[0]), math.ceil(input_size[1] / kernel_size[1])

    def get_output_size(self) -> int:
        output_shape = self.get_output_shape()
        return output_shape[0] * output_shape[1]

    def get_input_size(self) -> int:
        input_shape = self.get_true_input_shape()
        return input_shape[0] * input_shape[1]

    def get_weight_count(self) -> int:
        kernel_shape = self.get_true_kernel_shape()
        return kernel_shape[0] * kernel_shape[1]

    def get_total_nodes_in(self) -> int:
        return self.get_input_size()

    def get_total_nodes_out(self) -> int:
        return self.get_output_size() * self.get_filter_count()

    @staticmethod
    def get_bias_count() -> int:
        return 1

    def _forward(self, filter_index: int, inputs: NetworkBuffer) -> NetworkBuffer:
        weights = self.weights[filter_index]
        biases = self.biases[filter_index]

        # maybe this could be optimised, as we are writing an empty buffer to the gpu, instead of just making an empty gpu buffer
        output = NetworkBuffer(np.zeros(self.get_output_size()), self.get_output_size())

        filter_shape = self.get_true_kernel_shape()
        input_shape = self.get_true_input_shape()

        self.forward_core.forward(queue, self.get_output_shape(), None,
                                  inputs.get_as_buffer(), output.get_as_buffer(),
                                  weights.get_as_buffer(), biases.get_as_buffer(),
                                  np.int32(filter_shape[0]), np.int32(filter_shape[1]),
                                  np.int32(input_shape[0]), np.int32(input_shape[1]),
                                  np.int32(self.get_output_shape()[0])).wait()

        return output

    def forward(self, inputs: NetworkBuffer) -> tuple[Any]:
        return tuple(
            self._forward(filter_index, inputs) for filter_index in range(self.__filter_count)
        )


class FullyConnectedLayer(Layer):
    def __init__(self, input_size: int, output_size: int, activation: int):
        self.__input_size = input_size
        self.__output_size = output_size
        self.__activation = activation

        # todo - use different weight inits depending on activation
        self.weights = NetworkBuffer(relu_weight_init((input_size, output_size), input_size), self.get_weight_count())
        self.biases = NetworkBuffer(np.zeros(self.get_bias_count()), self.get_bias_count())

        self.forward_core = load_core("full_pop")

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size

    def get_weight_count(self):
        return self.__input_size * self.__output_size

    def get_bias_count(self):
        return self.__output_size

    def get_total_nodes_in(self) -> int:
        return self.get_input_size()

    def get_total_nodes_out(self) -> int:
        return self.get_output_size()

    def forward(self, inputs: NetworkBuffer) -> NetworkBuffer:
        output = NetworkBuffer(np.zeros(self.get_output_size()), self.get_output_size())
        unactivated = NetworkBuffer(np.empty(self.get_weight_count()), self.get_weight_count())

        self.forward_core.forward(queue, (self.get_input_size(), self.get_output_size()), None,
                                  inputs.get_as_buffer(), unactivated.get_as_buffer(),
                                  self.weights.get_as_buffer(), np.int32(self.get_input_size())
                                  ).wait()

        self.forward_core.reduce_outputs(queue, (self.get_output_size(),), None,
                                         unactivated.get_as_buffer(), output.get_as_buffer(),
                                         self.biases.get_as_buffer(), np.int32(self.get_input_size()),
                                         np.int32(self.get_output_size()), np.int32(self.__activation)
                                         ).wait()

        return output
