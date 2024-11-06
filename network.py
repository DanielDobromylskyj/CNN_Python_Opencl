import random
import time
import numpy as np
import math

import buffers
import layers
import activations
import file_api


class InvalidNetwork(Exception):
    pass


def validate_network_layout(layout):
    if len(layout) == 0:
        raise InvalidNetwork("Network Layout Contains No Layers")

    if len(layout) == 1:
        return

    previous_output_size = layout[0].get_total_nodes_out()

    if layout[0].get_total_nodes_in() <= 0:
        raise InvalidNetwork(f"Layer 1 has a invalid number of inputs '{layout[0].get_total_nodes_in()}'")

    for i, layer in enumerate(layout[1:]):
        if layer.get_total_nodes_in() <= 0:
            raise InvalidNetwork(f"Layer {i + 2} has a invalid number of inputs '{layout[0].get_total_nodes_in()}'")

        if previous_output_size <= 0:
            raise InvalidNetwork(f"Layer {i + 1} has a invalid number of outputs '{previous_output_size}'")

        if previous_output_size != layer.get_total_nodes_in():
            raise InvalidNetwork(
                f"Layer {i + 1} Outputs {previous_output_size} Values but Layer {i + 2} Takes {layer.get_total_nodes_in()}")

        previous_output_size = layer.get_total_nodes_out()


class Network:
    def __init__(self, layout: tuple, validate_network=True):
        self.layout = layout

        if validate_network:
            validate_network_layout(self.layout)

    def forward_pass(self, inputs, save_layer_data=False,
                     for_display=False):
        output_values = buffers.create_network_buffer_from_input(inputs)

        save_values = [] if not save_layer_data else [(output_values, None)]

        for layer in self.layout:
            output_values = layer.forward(output_values, save_layer_data)

            if for_display:
                save_values.append(output_values)

            if save_layer_data:  # this is useless right? (unpacking them)
                output_values, other_value = output_values
                save_values.append((output_values, other_value))

        if for_display or save_layer_data:
            return save_values

        if isinstance(output_values, buffers.NetworkBuffer):
            return output_values.get_as_array()
        else:
            return [output_value.get_as_array() for output_value in output_values]

    @staticmethod
    def __convert_to_gradients(data):
        if data is None:
            data = np.array([0], dtype=np.float32)

        return buffers.Gradients(data)

    @staticmethod
    def __ensure_is_network_buffer(data):
        if data is None:
            data = [0]

        if isinstance(data, buffers.NetworkBuffer):
            return data

        if isinstance(data, buffers.BufferList):
            return data

        if isinstance(data, list) or isinstance(data, tuple):
            data = np.array(data)

        if isinstance(data, np.ndarray):
            return buffers.NetworkBuffer(data, data.shape)

        raise ValueError(f"Could not ensure data is a network buffer or convert to one\nData:\n{data}")

    def backward_pass(self, inputs: np.ndarray, target: np.ndarray, learning_rate: float):
        data = self.forward_pass(inputs, save_layer_data=True)
        outputs = data[-1][0].get_as_array()

        error = target - outputs

        output_error_gradients = self.__convert_to_gradients(error)

        gradient_data = [[None, None] for j in range(len(self.layout))]

        for layer_index in range(len(self.layout) - 1, -1, -1):
            activated_inputs, unactivated_inputs = data[layer_index]  # Extract Inputs

            # Extract Outputs - Add one, as we have inputs in front of the data
            activated_outputs, unactivated_outputs = data[layer_index + 1]

            input_gradients, weight_gradients, bias_gradients = self.layout[layer_index].backward(
                self.__ensure_is_network_buffer(activated_inputs),
                self.__ensure_is_network_buffer(activated_outputs),
                self.__ensure_is_network_buffer(unactivated_outputs),
                output_error_gradients,
                learning_rate
            )

            # Prep for next layer
            output_error_gradients = input_gradients

            gradient_data[layer_index][0] = weight_gradients
            gradient_data[layer_index][1] = bias_gradients

        return gradient_data

    @staticmethod
    def __sum_array(data):
        summed_buffer_list = data[0]
        for item in data[1:]:
            if isinstance(item, buffers.BufferList):
                for buffer_index, buffer in enumerate(item):
                    summed_buffer_list[buffer_index] += buffer
            else:
                summed_buffer_list += item

        return summed_buffer_list

    def __condense_gradients(self, gradient_data, sub_index):
        total_gradients_to_sum = [[] for j in range(len(self.layout))]

        for sample in gradient_data:
            for layer_index, layer in enumerate(sample):
                data = layer[sub_index]
                total_gradients_to_sum[layer_index].append(data)

        return [
            self.__sum_array(layer_gradients)
            for layer_gradients in total_gradients_to_sum
        ]

    def compute_epoch(self, training_data, learning_rate: float):
        gradient_data = [
            self.backward_pass(sample, target, learning_rate)
            for sample, target in training_data
        ]

        weight_gradients = self.__condense_gradients(gradient_data, 0)
        bias_gradients = self.__condense_gradients(gradient_data, 1)

        for layer_index, layer in enumerate(self.layout):
            if np.any(np.isnan(weight_gradients[layer_index].get_as_array()) | np.isinf(
                    weight_gradients[layer_index].get_as_array())):
                raise ValueError("[FINAL CHECK] NaN or Inf Found In Weights (NaN, Inf): " + str(
                    np.any(np.isnan(weight_gradients[layer_index].get_as_array()))) + ", " + str(
                    np.any(np.isinf(weight_gradients[layer_index].get_as_array()))))

            if np.any(np.isnan(bias_gradients[layer_index].get_as_array()) | np.isinf(
                    bias_gradients[layer_index].get_as_array())):
                raise ValueError("[FINAL CHECK] NaN or Inf Found In Biases (NaN, Inf): " + str(
                    np.any(np.isnan(bias_gradients[layer_index].get_as_array()))) + ", " + str(
                    np.any(np.isinf(bias_gradients[layer_index].get_as_array()))))

            layer.apply_gradients(weight_gradients[layer_index], bias_gradients[layer_index], len(training_data))

    def test_network(self, test_data, tests=5, decimals=3):
        errors = []
        for j in range(tests):
            inputs, target = random.choice(test_data)
            outputs = self.forward_pass(inputs)
            errors.append(sum([
                abs(outputs[node] - target[node]) for node in range(len(outputs))
            ]))

        return round(sum(errors) / tests, decimals), round(min(errors), decimals), round(max(errors), decimals)

    def train(self, training_data, test_data, epochs, learning_rate, show_stats=True):
        start = time.time()

        for epoch in range(epochs):
            self.compute_epoch(training_data, learning_rate)

            elapsed = time.time() - start

            if show_stats:
                average_error, min_error, max_error = self.test_network(test_data)

                time_left_in_s = (elapsed / (epoch+1)) * (epochs-epoch)
                min_left = time_left_in_s // 60
                sec_left = time_left_in_s - (min_left * 60)

                percentage_bar_max_length = 50
                percentage_bar_done = math.floor(((epoch+1) / epochs) * percentage_bar_max_length)

                print(f"\r|{'#' * percentage_bar_done}{' ' * (percentage_bar_max_length - percentage_bar_done)}| Average Error: {average_error} | ETA: {round(min_left)}m {math.floor(sec_left)}s", end="", flush=True)


if __name__ == "__main__":
    import viewer

    net = Network((
        layers.ConvolutedLayer((100, 100), (5, 5), filter_count=5, colour_depth=3),
        layers.FullyConnectedLayer(20 * 20 * 5, 2, activations.Sigmoid)
    ))
    print("made network")

    rand_data = np.random.randn(30_000).astype(np.float32)
    v = viewer.viewer()

    l_rate = 0.001
    train_data = [(rand_data, np.array([0, 1])), (rand_data, np.array([0, 1]))]
    test_data = [(rand_data, np.array([0, 1]))]

    net.train(train_data, test_data, 50, l_rate)
