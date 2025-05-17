from __future__ import annotations
from typing import List, Any, Tuple

import pyopencl as cl
import numpy as np
import math
import base64

from buffers import *
from buffers import BufferList

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


def load_core(core_element):
    """ Loads a GPU program """
    with open(f"./core/{core_element}.txt", "r") as f:
        return cl.Program(ctx, f.read()).build()


def mul(sizes):
    """ Helper Function """
    total = 1
    for size in sizes:
        total *= size

    return total


def relu_weight_init(size, fan_in, dtype=np.float32):
    """ Creates a random uniform weight initialization """
    assert fan_in > 0, "fan_in must be a positive number"

    random_values = np.random.randn(mul(size))
    random_values = np.clip(random_values, -10, 10)
    return random_values.astype(dtype) * np.sqrt(2.0 / fan_in)


def sigmoid_weight_init(size, fan_in, dtype=np.float32):
    """ Creates a random uniform weight initialization """
    assert fan_in > 0, "fan_in must be a positive number"

    random_values = np.random.randn(np.prod(size))
    random_values = np.clip(random_values, -10, 10)
    return random_values.astype(dtype) * np.sqrt(2.0 / fan_in)


class ADAM:  # Adaptive Momentum Estimation
    def __init__(self, shape, beta1=0.9, beta2=0.999):
        """ Adaptive Momentum Estimation (Algorithm) """
        self.beta_1 = beta1  # First moment decay
        self.beta_2 = beta2  # Second moment decay
        self.epsilon = 1e-7  # Small constant to avoid division by zero
        self.time_step = 0

        self.m = np.zeros(shape, dtype=np.float32)  # First moment (m_t)
        self.v = np.zeros(shape, dtype=np.float32)  # Second moment (v_t)

    def optimise(self, parameters, gradients):
        """ Performs calculations to control the parameters of the network"""
        gradients = gradients.get_as_array()

        for param in range(len(parameters)):
            g_t = gradients[param]  # Gradient for parameter at time t

            # Update biased first moment estimate (m_t)
            self.m[param] = self.beta_1 * self.m[param] + (1 - self.beta_1) * g_t

            # Update biased second moment estimate (v_t)
            self.v[param] = self.beta_2 * self.v[param] + (1 - self.beta_2) * (g_t ** 2)

            # Compute bias-corrected estimates
            m_hat = self.m[param] / (1 - self.beta_1 ** self.time_step)
            v_hat = self.v[param] / (1 - self.beta_2 ** self.time_step)

            # Update parameter using ADAM formula
            parameters[param] += m_hat / (np.sqrt(v_hat) + self.epsilon)

        return parameters

    def step(self):
        self.time_step += 1


class LayerCodes:
    def __init__(self):
        """ Shorthand codes for layer types (Used for storage)"""
        self.__code_to_layer = {
            "FP": FullyConnectedLayer,
            "CV": ConvolutedLayer,
            "MP": MaxPooling
        }

        self.__layer_to_code = {v: k for k, v in self.__code_to_layer.items()}

    def __getitem__(self, item):
        if type(item) is bytes:
            return self.__code_to_layer[item.decode()]
        return self.__layer_to_code[item.__class__].encode()


class Layer:
    def forward(self, inputs: NetworkBuffer, save_layer_data=False):
        """ Perform a forward pass over the layer """
        raise NotImplementedError

    def backward(self, inputs: NetworkBuffer, outputs_activated: NetworkBuffer, outputs_unactivated: NetworkBuffer,
                 output_error_gradients: Gradients,
                 learning_rate: float) -> tuple[Gradients, Gradients, Gradients]:
        """ Calculate the gradients and errors for weights, biases and the 'next' layer """
        raise NotImplementedError

    def apply_gradients(self, weight_gradients: Gradients | BufferList, bias_gradients: Gradients | BufferList,
                        count: int) -> None:
        """ Applies the gradients we have calculated """
        raise NotImplementedError

    def get_total_nodes_in(self):
        raise NotImplementedError("Layer does not have a implemented 'get_total_nodes_in' method")

    def get_total_nodes_out(self):
        raise NotImplementedError("Layer does not have a implemented 'get_total_nodes_out' method")

    def serialize(self):
        raise NotImplementedError

    @staticmethod
    def deserialize(data):
        raise NotImplementedError


class MaxPooling(Layer):
    def __init__(self, input_size, size):
        self.__input_size = input_size
        self.__size = size

    def get_total_nodes_in(self):
        raise NotImplementedError("Layer does not have a implemented 'get_total_nodes_in' method")

    def get_total_nodes_out(self):
        raise NotImplementedError("Layer does not have a implemented 'get_total_nodes_out' method")

    def apply_gradients(self, weight_gradients: Gradients | BufferList, bias_gradients: Gradients | BufferList,
                        count: int) -> None:
        pass

    def serialize(self):
        return f"{self.__input_size}..{self.__size}".encode()

    @staticmethod
    def deserialize(data):
        input_size, size = data.decode().split("..")

        return MaxPooling(eval(input_size), eval(size))


class ConvolutedLayer(Layer):
    def __init__(self, input_size: tuple[int, int], kernel_size: tuple[int, int], filter_count: int,
                 colour_depth: int = 1, stride: int = 1, loading: bool = False, testing=None):  # todo - allow for sigmoid / other activations
        self.__filter_count = filter_count
        self.__input_size = input_size
        self.__kernel_size = kernel_size
        self.__colour_depth = colour_depth
        self.__stride = stride

        self.forward_core = load_core("kernel")

        # Don't spend computing power generating weights to just overwrite them when loading.
        # todo - clean up this init data ffs it's a mess
        self.weights = [
            NetworkBuffer(relu_weight_init((self.get_weight_count(),), self.get_weight_count()) if not testing else
                                    np.full(self.get_weight_count(), testing[0], dtype=np.float32),
                          self.get_weight_count())
            for i in range(self.__filter_count)
        ] if loading is False else None

        self.biases = [
            NetworkBuffer(
                np.zeros((1,), dtype=np.float32) if not testing else np.full(self.get_bias_count(), testing[1], dtype=np.float32),
                self.get_bias_count()) if loading is False else None
            for i in range(self.__filter_count)
        ] if loading is False else None

    def get_filter_count(self):
        """ Returns the number of filters / kernels"""
        return self.__filter_count

    def get_true_kernel_shape(self) -> tuple[int, int]:
        """ Returns the shape of the true kernel """
        return self.__kernel_size[0] * self.__colour_depth, self.__kernel_size[1]

    def get_true_input_shape(self) -> tuple[int, int]:
        """ Returns the shape of the true input """
        return self.__input_size[0] * self.__colour_depth, self.__input_size[1]

    def get_output_shape(self) -> tuple[int, int]:
        """ Returns the shape of the output """
        kernel_size = self.get_true_kernel_shape()
        input_size = self.get_true_input_shape()

        return math.ceil((input_size[0] - kernel_size[0]) / self.__stride), math.ceil((input_size[1] - kernel_size[1]) / self.__stride)

    def get_output_size(self) -> int:
        """ Returns the size of the output - Used for network validation """
        output_shape = self.get_output_shape()
        return int(output_shape[0] * output_shape[1])

    def get_input_size(self) -> int:
        """ Returns the size of the input - Used for network validation"""
        input_shape = self.get_true_input_shape()
        return input_shape[0] * input_shape[1]

    def get_weight_count(self) -> int:
        """ Returns the number of weights """
        kernel_shape = self.get_true_kernel_shape()
        return kernel_shape[0] * kernel_shape[1]

    def get_total_nodes_in(self) -> int:
        """ Returns the total number of nodes used for input - Used for network validation"""
        return self.get_input_size()

    def get_total_nodes_out(self) -> int:
        """ Returns the total number of nodes used for output - Used for network validation"""
        return self.get_output_size() * self.get_filter_count()

    @staticmethod
    def get_bias_count() -> int:
        """ Returns the number of biases. Which is always 1. So, boring function 'return 1'"""
        return 1

    def _forward(self, filter_index: int, inputs: NetworkBuffer) -> NetworkBuffer:
        """ Performs a forward pass over the layer, on a single filter / kernel"""
        weights = self.weights[filter_index]
        biases = self.biases[filter_index]

        # maybe this could be optimised, as we are writing an empty buffer to the gpu, instead of just making an empty gpu buffer
        output = NetworkBuffer(np.zeros(self.get_output_size(), dtype=np.float32), (self.get_output_size(),))

        filter_shape = self.get_true_kernel_shape()
        input_shape = self.get_true_input_shape()

        self.forward_core.forward(queue, self.get_output_shape(), None,
                                  inputs.get_as_buffer(), output.get_as_buffer(),
                                  weights.get_as_buffer(), biases.get_as_buffer(),
                                  np.int32(filter_shape[0]), np.int32(filter_shape[1]),
                                  np.int32(input_shape[0]), np.int32(input_shape[1]),
                                  np.int32(self.get_output_shape()[0]), np.int32(self.__stride)).wait()

        return output

    def forward(self, inputs: NetworkBuffer, save_layer_data=False) -> tuple[BufferList, None] | BufferList:
        """ Loops over all kernels and runs a forward pass over the data """
        data = BufferList(tuple(
            self._forward(filter_index, inputs) for filter_index in range(self.__filter_count)
        ))

        if save_layer_data:
            return data, None
        return data

    def _backward(self, core, filter_index, inputs: NetworkBuffer, outputs_activated: NetworkBuffer,
                  outputs_unactivated: NetworkBuffer, output_error_gradients: Gradients,
                  learning_rate: float) -> tuple[Gradients, Gradients, Gradients]:
        """ Performs a single backward pass over the data with a single kernel"""

        input_gradients = Gradients(np.zeros(self.get_input_size(), dtype=np.float32))
        weight_gradients_unreduced = Gradients(np.zeros(self.get_input_size(), dtype=np.float32))
        weight_gradients = Gradients(np.zeros(self.get_weight_count(), dtype=np.float32))
        bias_gradients = Gradients(np.zeros(self.get_bias_count(), dtype=np.float32))

        filter_shape = self.get_true_kernel_shape()
        input_shape = self.get_true_input_shape()
        output_shape = self.get_output_shape()

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
                    np.int32(self.__stride),
                    np.float32(learning_rate)
                    ).wait()

        if np.any(np.isnan(weight_gradients_unreduced.get_as_array()) | np.isinf(
                weight_gradients_unreduced.get_as_array())):
            raise ValueError("[KERNEL][NO SUM] NaN or Inf Found In Weights (NaN, Inf): " + str(
                np.any(np.isnan(weight_gradients_unreduced.get_as_array()))) + ", " + str(
                np.any(np.isinf(weight_gradients_unreduced.get_as_array()))))

        core.sum_gradients(
            queue, filter_shape, None,
            weight_gradients_unreduced.get_as_buffer(),
            weight_gradients.get_as_buffer(),
            np.int32(filter_shape[0]),
            np.int32(filter_shape[1]),
            np.int32(input_shape[0]),
            np.int32(input_shape[1]),
            np.int32(self.get_output_shape()[0] // filter_shape[0]),
            np.int32(self.get_output_shape()[1] // filter_shape[1])
        ).wait()

        queue.finish()

        if np.any(np.isnan(weight_gradients.get_as_array()) | np.isinf(
                weight_gradients.get_as_array())):
            raise ValueError("[KERNEL][AFTER SUM] NaN or Inf Found In Weights (NaN, Inf): " + str(
                np.any(np.isnan(weight_gradients.get_as_array()))) + ", " + str(
                np.any(np.isinf(weight_gradients.get_as_array()))))

        return input_gradients, weight_gradients, bias_gradients

    def backward(self, inputs: NetworkBuffer, outputs_activated: NetworkBuffer,
                 outputs_unactivated: NetworkBuffer, output_error_gradients: Gradients,
                 learning_rate: float) -> tuple[Gradients, Gradients, Gradients]:
        """ Loops over all kernels and performs a backward pass on them """
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

    def apply_gradients(self, weight_gradients: BufferList, bias_gradients: BufferList, count: int) -> None:  # todo - add ADAM
        """ Applies the gradients to the weights and biases, This could be made to use ADAM as well"""
        for filter_index, weight_gradient in enumerate(weight_gradients):
            self.weights[filter_index] += weight_gradient / count

        for filter_index, bias_gradient in enumerate(bias_gradients):
            self.biases[filter_index] += bias_gradient / count

    def serialize(self):
        """ Compress all layer data to a stringof bytes"""
        return b",".join([
            base64.b64encode(
                weight.get_as_array().tobytes()
            ) for weight in self.weights
        ]) + b".." + b",".join([
            base64.b64encode(
                bias.get_as_array().tobytes()
            ) for bias in self.biases
        ]) + f"..{self.__filter_count}..{self.__input_size}..{self.__kernel_size}..{self.__colour_depth}..{self.__stride}".encode()

    @staticmethod
    def deserialize(data):
        """ Turn a string of bytes into a layer instance"""
        weights, biases, filter_count, input_size, kernel_size, colour_depth, stride = data.split(b"..")

        layer = ConvolutedLayer(
            eval(input_size.decode()),
            eval(kernel_size.decode()),
            int(filter_count.decode()),
            int(colour_depth.decode()),
            int(stride.decode()),
            loading=True
        )

        layer.weights = [
            create_network_buffer_from_input(np.frombuffer(
                base64.b64decode(weight_bytes), dtype=np.float32
            )) for weight_bytes in weights.split(b",")
        ]

        layer.biases = [
            create_network_buffer_from_input(np.frombuffer(
                base64.b64decode(bias_bytes), dtype=np.float32
            )) for bias_bytes in biases.split(b",")
        ]

        return layer


class FullyConnectedLayer(Layer):
    def __init__(self, input_size: int, output_size: int, activation: int, loading: bool = False, testing=None, ADAM_data=None):
        self.__input_size = input_size
        self.__output_size = output_size
        self.__activation = activation

        self.weights = NetworkBuffer(
            (
                relu_weight_init((input_size, output_size), input_size) if activation == 1 else
                sigmoid_weight_init((input_size, output_size), input_size)
            ) if not testing else np.full((input_size * output_size,), testing[0], dtype=np.float32),
            self.get_weight_count()) if loading is False else None

        self.biases = NetworkBuffer(
            np.zeros(self.get_bias_count()) if not testing else np.full(self.get_bias_count(), testing[1], dtype=np.float32),
            self.get_bias_count()
        ) if loading is False else None

        if ADAM_data:
            beta1, beta2 = ADAM_data
            self.optimisers_weights = ADAM(self.get_weight_count(), beta1, beta2)
            self.optimisers_biases = ADAM(self.get_bias_count(), beta1, beta2)
        else:
            self.optimisers_weights = None
            self.optimisers_biases = None

        self.forward_core = load_core("full_pop")

    def get_input_size(self):
        """ Returns the size of inputs """
        return self.__input_size

    def get_output_size(self):
        """ Returns the size of outputs """
        return self.__output_size

    def get_weight_count(self):
        """ Returns the number of weights """
        return self.__input_size * self.__output_size

    def get_bias_count(self):
        """ Returns the number of biases """
        return self.__output_size

    def get_total_nodes_in(self) -> int:
        """ Returns the total number of nodes in"""
        return self.get_input_size()

    def get_total_nodes_out(self) -> int:
        """ Returns the total number of nodes out"""
        return self.get_output_size()

    def forward(self, inputs: NetworkBuffer, save_layer_data=False) -> NetworkBuffer | tuple[
        NetworkBuffer, NetworkBuffer]:
        """ Performs a forward pass over all the input data """

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
        """ Calculates gradients and errors using backpropagation"""
        core = load_core("training/full_pop")

        # Create Gradients
        weight_gradients = Gradients(np.zeros(self.get_weight_count(), dtype=np.float32))
        bias_gradients_unreduced = Gradients(np.zeros(self.get_weight_count(), dtype=np.float32))
        bias_gradients = Gradients(np.zeros(self.get_bias_count(), dtype=np.float32))
        input_gradients_unreduced = Gradients(np.zeros(self.get_weight_count(), dtype=np.float32))
        input_gradients = Gradients(np.zeros(self.get_input_size(), dtype=np.float32))

        # Step 1/3 - Perform calculations
        core.backwards(queue, (self.get_input_size(), self.get_output_size()), None,
                       inputs.get_as_buffer(), outputs_activated.get_as_buffer(),
                       outputs_unactivated.get_as_buffer(), self.weights.get_as_buffer(),
                       self.biases.get_as_buffer(), output_error_gradients.get_as_buffer(),
                       input_gradients_unreduced.get_as_buffer(), weight_gradients.get_as_buffer(),
                       bias_gradients_unreduced.get_as_buffer(), np.int32(self.get_input_size()),
                       np.int32(self.get_output_size()), np.int32(self.__activation), np.float32(learning_rate)
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

        if np.any(np.isnan(weight_gradients.get_as_array()) | np.isinf(
                weight_gradients.get_as_array())):
            raise ValueError("[FULL POP] NaN or Inf Found In Weights (NaN, Inf): " + str(
                np.any(np.isnan(weight_gradients.get_as_array()))) + ", " + str(
                np.any(np.isinf(weight_gradients.get_as_array()))))

        return input_gradients, weight_gradients, bias_gradients

    def apply_gradients(self, weight_gradients: Gradients, bias_gradients: Gradients, count: int) -> None:
        """ Applies gradients to weights and biases, using adam if enabled """
        if not self.optimisers_weights:
            self.weights += weight_gradients / count
            self.biases += bias_gradients / count

        else:
            weights = self.weights.get_as_array()
            biases = self.biases.get_as_array()

            self.optimisers_weights.step()
            adjusted_weights = self.optimisers_weights.optimise(weights, weight_gradients)
            self.weights = NetworkBuffer(adjusted_weights, adjusted_weights.shape)

            self.optimisers_biases.step()
            adjusted_biases = self.optimisers_biases.optimise(biases, bias_gradients)
            self.biases = NetworkBuffer(adjusted_biases, adjusted_biases.shape)

    def serialize(self):
        """ Returns a serialized version of the layers in bytes"""
        return base64.b64encode(self.weights.get_as_array().tobytes()) + b".." + base64.b64encode(
            self.biases.get_as_array().tobytes()) + f"..{self.__input_size}..{self.__output_size}..{self.__activation}".encode()

    @staticmethod
    def deserialize(data):
        """ Converts bytes into a layer instance"""
        weights, biases, input_size, output_size, activation = data.split(b"..")

        layer = FullyConnectedLayer(
            eval(input_size.decode()),
            eval(output_size.decode()),
            int(activation.decode()),
            loading=True
        )

        layer.weights = create_network_buffer_from_input(
            np.frombuffer(base64.b64decode(weights), dtype=np.float32)
        )

        layer.biases = create_network_buffer_from_input(
            np.frombuffer(base64.b64decode(biases), dtype=np.float32)
        )

        return layer
