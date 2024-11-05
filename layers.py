from __future__ import annotations

from typing import List, Any

import pyopencl as cl
import numpy as np
import math

from buffers import *

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
    def forward(self, inputs: NetworkBuffer, save_layer_data=False):
        """ Perform a forward pass over the layer """
        raise NotImplementedError

    def backward(self, inputs: NetworkBuffer, outputs_activated: NetworkBuffer, outputs_unactivated: NetworkBuffer,
                 output_error_gradients: Gradients,
                 learning_rate: float) -> tuple[Gradients, Gradients, Gradients]:
        """ Calculate the gradients and errors for weights, biases and the 'next' layer """
        raise NotImplementedError

    def apply_gradients(self, weight_gradients: Gradients | BufferList, bias_gradients: Gradients | BufferList) -> None:
        """ Applies the gradients we have calculated """
        raise NotImplementedError

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
            NetworkBuffer(relu_weight_init((self.get_weight_count(),), self.get_weight_count()),
                          self.get_weight_count())
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
        output = NetworkBuffer(np.zeros(self.get_output_size(), dtype=np.float32), self.get_output_size())

        filter_shape = self.get_true_kernel_shape()
        input_shape = self.get_true_input_shape()

        self.forward_core.forward(queue, self.get_output_shape(), None,
                                  inputs.get_as_buffer(), output.get_as_buffer(),
                                  weights.get_as_buffer(), biases.get_as_buffer(),
                                  np.int32(filter_shape[0]), np.int32(filter_shape[1]),
                                  np.int32(input_shape[0]), np.int32(input_shape[1]),
                                  np.int32(self.get_output_shape()[0])).wait()

        return output

    def forward(self, inputs: NetworkBuffer, save_layer_data=False) -> tuple[Any] | tuple[tuple[Any], None]:
        data = BufferList(tuple(
            self._forward(filter_index, inputs) for filter_index in range(self.__filter_count)
        ))

        if save_layer_data:
            return data, None
        return data

    def _backward(self, core, filter_index, inputs: NetworkBuffer, outputs_activated: NetworkBuffer,
                  outputs_unactivated: NetworkBuffer, output_error_gradients: Gradients,
                  learning_rate: float) -> tuple[Gradients, Gradients, Gradients]:
        input_gradients = Gradients(np.empty(self.get_input_size(), dtype=np.float32))
        weight_gradients_unreduced = Gradients(np.empty(self.get_input_size(), dtype=np.float32))
        weight_gradients = Gradients(np.empty(self.get_weight_count(), dtype=np.float32))
        bias_gradients = Gradients(np.empty(self.get_bias_count(), dtype=np.float32))

        filter_shape = self.get_true_kernel_shape()
        input_shape = self.get_true_input_shape()
        output_shape = self.get_output_shape()

        queue.finish()

        core.filter(queue, output_shape, None,
                    input_gradients.get_as_buffer(),
                    self.weights[filter_index].get_as_buffer(),
                    self.biases[filter_index].get_as_buffer(),
                    output_error_gradients.get_as_buffer(),
                    inputs.get_as_buffer(),
                    outputs_activated.get_as_buffer(),
                    weight_gradients_unreduced.get_as_buffer(),
                    bias_gradients.get_as_buffer(),
                    np.int32(input_shape[0]),
                    np.int32(input_shape[1]),
                    np.int32(filter_shape[0]),
                    np.int32(filter_shape[1]),
                    np.int32(self.get_output_shape()[0]),
                    np.float32(learning_rate)
                    ).wait()

        core.sum_gradients(
            queue, filter_shape, None,
            weight_gradients_unreduced.get_as_buffer(),
            weight_gradients.get_as_buffer(),
            np.int32(filter_shape[0]),
            np.int32(filter_shape[1]),
            np.int32(input_shape[0]),
            np.int32(input_shape[1]),
            np.int32(self.get_output_shape()[0]),
            np.int32(self.get_output_shape()[1])
        ).wait()

        queue.finish()

        return input_gradients, weight_gradients, bias_gradients

    def backward(self, inputs: NetworkBuffer, outputs_activated: NetworkBuffer,
                 outputs_unactivated: NetworkBuffer, output_error_gradients: Gradients,
                 learning_rate: float) -> tuple[Gradients, Gradients, Gradients]:
        core = load_core("training/kernel")

        # Make it an index able object
        output_error_gradients = convert_gradients_to_buffer_list(output_error_gradients, 1)

        return rearrange_feature_map_output(tuple(
            self._backward(
                core, filter_index,
                inputs,
                outputs_activated.get_network_buffer(filter_index),
                None,
                output_error_gradients.get_network_buffer(filter_index),
                learning_rate
            ) for filter_index in range(self.__filter_count)
        ))

    def apply_gradients(self, weight_gradients: BufferList, bias_gradients: BufferList) -> None:
        for filter_index, weight_gradient in enumerate(weight_gradients):
            self.weights[filter_index] += weight_gradient

        for filter_index, bias_gradient in enumerate(bias_gradients):
            self.biases[filter_index] += bias_gradient


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

    def forward(self, inputs: NetworkBuffer, save_layer_data=False) -> NetworkBuffer | tuple[
        NetworkBuffer, NetworkBuffer]:
        output = NetworkBuffer(np.zeros(self.get_output_size(), dtype=np.float32), self.get_output_size())
        unreduced_outputs = NetworkBuffer(np.empty(self.get_weight_count(), dtype=np.float32), self.get_weight_count())
        unactivated_outputs = NetworkBuffer(np.empty(self.get_output_size(), dtype=np.float32), self.get_output_size())

        self.forward_core.forward(queue, (self.get_input_size(), self.get_output_size()), None,
                                  inputs.get_as_buffer(), unreduced_outputs.get_as_buffer(),
                                  self.weights.get_as_buffer(), np.int32(self.get_input_size())
                                  ).wait()

        self.forward_core.reduce_outputs(queue, (self.get_output_size(),), None,
                                         unreduced_outputs.get_as_buffer(), output.get_as_buffer(),
                                         unactivated_outputs.get_as_buffer(), self.biases.get_as_buffer(),
                                         np.int32(self.get_input_size()), np.int32(self.get_output_size()),
                                         np.int32(self.__activation)
                                         ).wait()

        if save_layer_data:
            return output, unactivated_outputs

        return output

    def backward(self, inputs: NetworkBuffer, outputs_activated: NetworkBuffer,
                 outputs_unactivated: NetworkBuffer, output_error_gradients: Gradients,
                 learning_rate: float) -> tuple[Gradients, Gradients, Gradients]:
        core = load_core("training/full_pop")

        # Create Gradients
        weight_gradients = Gradients(np.empty(self.get_weight_count(), dtype=np.float32))
        bias_gradients_unreduced = Gradients(np.empty(self.get_weight_count(), dtype=np.float32))
        bias_gradients = Gradients(np.empty(self.get_bias_count(), dtype=np.float32))
        input_gradients_unreduced = Gradients(np.empty(self.get_weight_count(), dtype=np.float32))
        input_gradients = Gradients(np.empty(self.get_input_size(), dtype=np.float32))

        # Step 1/3 - Perform calculations
        core.backwards(queue, (self.get_input_size(), self.get_output_size()), None,
                       inputs.get_as_buffer(), outputs_activated.get_as_buffer(),
                       outputs_unactivated.get_as_buffer(), self.weights.get_as_buffer(),
                       self.biases.get_as_buffer(), output_error_gradients.get_as_buffer(),
                       input_gradients_unreduced.get_as_buffer(), weight_gradients.get_as_buffer(),
                       bias_gradients_unreduced.get_as_buffer(), np.int32(self.get_input_size()),
                       np.int32(self.__activation)
                       ).wait()

        # Step 2/3 - Reduce Input Gradients
        core.reduce_input_error_gradients(queue, (self.get_input_size(),), None,
                                          input_gradients_unreduced.get_as_buffer(),
                                          input_gradients.get_as_buffer(),
                                          np.int32(self.get_input_size()),
                                          np.int32(self.get_output_size())
                                          ).wait()

        # Step 3/3 - Reduce Bias Gradients
        core.reduce_bias_gradients(queue, (self.get_output_size(),), None,
                                   bias_gradients_unreduced.get_as_buffer(),
                                   bias_gradients.get_as_buffer(),
                                   np.int32(self.get_input_size()),
                                   np.int32(self.get_output_size())
                                   ).wait()

        return input_gradients, weight_gradients, bias_gradients

    def apply_gradients(self, weight_gradients: Gradients, bias_gradients: Gradients) -> None:
        self.weights += weight_gradients
        self.biases += bias_gradients
