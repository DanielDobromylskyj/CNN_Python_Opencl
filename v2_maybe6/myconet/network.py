import pyopencl as cl
import numpy as np
import os

from .core.load import load_kernel, load_training_kernel
from . import file_api, buffer
from .layer import loader

from .optimisers import standard


class InvalidNetwork(Exception):
    pass



class OpenCL_Instance:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)



class Network:
    def __init__(self, layout: tuple, verify=True, cl_instance=None):
        self.cl = OpenCL_Instance() if not cl_instance else cl_instance
        self.__kernels = {}
        self.layout = layout

        if verify:
            self.validate_layout()

        self.__ready_kernels()
        self.__optimiser = standard.Optimiser(self)


    def __ready_kernels(self, load_training_kernels=False):
        kernels_previously_loaded = self.__kernels != {}
        self.__kernels = {}

        for layer in self.layout:
            if layer.__class__.__name__ not in self.__kernels:
                self.__kernels[layer.__class__.__name__] = (
                    load_kernel(self.cl, layer.get_kernel_name()),
                    load_training_kernel(self.cl, layer.get_kernel_name()) if load_training_kernels else None
                )

            layer.set_kernels(self.cl, self.__kernels[layer.__class__.__name__])

            if not kernels_previously_loaded:
                layer.init_values()

    def validate_layout(self):
        for i in range(len(self.layout) - 1):
            nodes_out = self.layout[i].get_node_count()[1]
            nodes_in = self.layout[i + 1].get_node_count()[0]
            if nodes_in != nodes_out:
                raise InvalidNetwork(
                    f"Layout Invalid -> Layer {i+1} outputs {nodes_out}, Yet Layer {i+2} takes {nodes_in} inputs."
                )


    def capture_forward(self, input_buffer: buffer.NetworkBuffer):  # Use for training, returns extra data
        extra_data = []
        node_values = [input_buffer.get_as_array()]

        for layer in self.layout:
            values = layer.forward_train(input_buffer)
            input_buffer = values[0]

            extra_data.append(values)
            node_values.append(values[0].get_as_array())

        return input_buffer.get_as_array(), extra_data, node_values


    def forward(self, inputs):  # Not for training. Optimised for speed, use capture_forward(input)
        input_buffer: buffer.NetworkBuffer = buffer.create_network_buffer_from_input(self.cl, inputs)

        for layer in self.layout:
            input_buffer = layer.forward(input_buffer)
        
        return input_buffer.get_as_array()

    @staticmethod
    def __average_grads(data):
        sample_count = len(data)
        layer_count = len(data[0])

        averaged = data[0]

        for i in range(sample_count-1):
            layers = data[i+1]

            for layerIndex, layer in enumerate(layers):
                averaged[layerIndex][0] += layer[0]
                averaged[layerIndex][1] += layer[1]

        for layerIndex in range(layer_count):
            averaged[layerIndex][0] /= sample_count
            averaged[layerIndex][1] /= sample_count

        return averaged

    def backward(self, inputs: np.ndarray, target: np.ndarray, learning_rate: float):
        inputs = buffer.create_network_buffer_from_input(self.cl, inputs)

        output, backprop_data, layer_node_values = self.capture_forward(inputs)

        layer_error = target - output
        error_gradient = buffer.NetworkBuffer(self.cl, layer_error.astype(np.float32), output.shape)

        backprop_gradients = []

        for i in range(len(self.layout)):
            layer_index = len(self.layout) - i - 1
            layer = self.layout[layer_index]

            next_error_gradient, weight_gradients, bias_gradients = layer.backward(
                layer_node_values[layer_index],
                error_gradient,
                backprop_data[layer_index],
                learning_rate
            )

            error_gradient = next_error_gradient
            backprop_gradients.append([
                weight_gradients.get_and_release(),  # Store the weights on the CPU side only to save VRAM
                bias_gradients.get_and_release()
            ])

        return backprop_gradients

    def set_optimiser(self, optimiser):
        self.__optimiser = optimiser(self)

    def apply_gradients(self, gradients):
        self.__optimiser.apply_gradients(gradients)

    def score(self, inputs, targets):
        outputs = self.forward(inputs)

        return sum([
            abs(error) for error in outputs - targets
        ])


    def train(self, training_data, validation_data, epoches, learning_rate):  # todo - add validation
        self.__ready_kernels(load_training_kernels=True)

        for epoch in range(epoches):
            gradients = [
                self.backward(sample, sample.output, learning_rate)
                for sample in training_data
            ]

            averaged_gradients = self.__average_grads(gradients)
            self.apply_gradients(averaged_gradients)


    def save(self, path):
        open(path, "w").close()  # truncate

        with open(path, 'ab') as f:
            file_api.encode_number(len(self.layout), f)

            for layer in self.layout:
                file_api.encode_number(
                    loader.layer_to_code(layer), f
                )
                layer.save(f)


    @staticmethod
    def load(path):
        cl_instance = OpenCL_Instance()

        with open(path, 'rb') as f:
            layer_count = file_api.decode_int(f)

            layout = tuple([
                loader.code_to_layer(file_api.decode_int(f)).load(cl_instance, f)
                for _ in range(layer_count)
            ])

        return Network(layout, cl_instance=cl_instance)
