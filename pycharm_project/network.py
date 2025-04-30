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
    """ Ensures that the network layout is valid, and that the layer sizes are not mismatched"""
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

        self.__current_epoch = 0
        self.__max_epochs = 0

        self.__current_sample_count = 0
        self.__max_sample_count = 0

        self.__current_sample_progress = 0
        self.__max_sample_progress = 0

        self.__epoch_error = 0
        self.__epoch_errors = []

        if validate_network:
            validate_network_layout(self.layout)

    def get_network_mem_size(self):
        size_bytes = 0
        for layer in self.layout:
            if isinstance(layer, layers.FullyConnectedLayer) or isinstance(layer, layers.ConvolutedLayer):
                size_bytes += layer.get_weight_count() * 4  # float32
                size_bytes += layer.get_bias_count() * 4 # float32
                size_bytes += layer.get_output_size() * 4 # float32
                size_bytes += layer.get_input_size() * 4 # float32

        return size_bytes

    def get_gpu_buffer_size(self):
        size_bytes = 0
        for layer in self.layout:
            if isinstance(layer, layers.FullyConnectedLayer) or isinstance(layer, layers.ConvolutedLayer):
                size_bytes += layer.get_weight_count() * 4  # float32
                size_bytes += layer.get_bias_count() * 4 # float32
                size_bytes += layer.get_output_size() * 4 # float32

        return size_bytes


    def get_neuron_count(self) -> str:
        neurons = 0
        for layer in self.layout:
            if isinstance(layer, layers.FullyConnectedLayer) or isinstance(layer, layers.ConvolutedLayer):
                neurons += layer.get_weight_count()

        if len(str(neurons)) > 5:
            return  f"{neurons:.2e}"


        return str(neurons)

    def get_bias_count(self) -> str:
        bias = 0
        for layer in self.layout:
            if isinstance(layer, layers.FullyConnectedLayer) or isinstance(layer, layers.ConvolutedLayer):
                bias += layer.get_bias_count()

        if len(str(bias)) > 5:
            return  f"{bias:.2e}"


        return str(bias)


    def forward_pass(self, inputs, save_layer_data=False,
                     for_display=False):
        """ Performs a forward pass through the network."""
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

    def backward_pass(self, inputs: np.ndarray, target: np.ndarray, learning_rate: float, index=None):
        if index:
            self.__current_sample_count = index

        data = self.forward_pass(inputs, save_layer_data=True)
        """ Performs a backward pass though the network for gradient calculations """
        outputs = data[-1][0].get_as_array()

        error = target - outputs

        self.__epoch_error += sum([abs(err) for err in error]) / len(error)

        output_error_gradients = self.__convert_to_gradients(error.astype(np.float32))

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

    def get_error_history(self):
        return self.__epoch_errors

    def compute_epoch(self, training_data, learning_rate: float):
        """ Performs a training cycle for each piece of training data"""
        self.__epoch_error = 0

        gradient_data = [
            self.backward_pass(sample, target, learning_rate, index=i)
            for i, (sample, target) in enumerate(training_data)
        ]

        self.__epoch_errors.append(self.__epoch_error)

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

    def get_display_data(self):
        return (self.__current_epoch / self.__max_epochs), (self.__current_sample_count / self.__max_sample_count), 0

    def get_extra_display_data(self):
        return self.__current_epoch, self.__max_epochs, self.__current_sample_count, self.__max_sample_count,


    def train(self, training_data, test_data, epochs, learning_rate, show_stats=True):
        start = time.time()

        self.__max_epochs = epochs

        for epoch in range(epochs):
            self.__current_epoch = epoch
            self.__max_sample_count = len(training_data)

            self.compute_epoch(training_data, learning_rate)

            elapsed = time.time() - start

            if show_stats:
                average_error, min_error, max_error = self.test_network(test_data)

                time_left_in_s = (elapsed / (epoch+1)) * (epochs-epoch)
                min_left = time_left_in_s // 60
                sec_left = time_left_in_s - (min_left * 60)

                percentage_bar_max_length = 50
                percentage_bar_done = math.floor(((epoch+1) / epochs) * percentage_bar_max_length)

                print(f"\r|{'#' * percentage_bar_done}{' ' * (percentage_bar_max_length - percentage_bar_done)}| Average Error: {average_error} | Min/Max: {min_error}/{max_error} | ETA: {round(min_left)}m {math.floor(sec_left)}s", end="", flush=True)

    def save(self, path):
        file = file_api.File(path)

        layer_codes = layers.LayerCodes()
        for layer_id, layer in enumerate(self.layout):
            file.segments[f"layer_{layer_id}"] = layer_codes[layer] + layer.serialize()
        file.write()

    def debug_dump(self):
        """ Outputs useful information about the network incase of a problem """
        print(">>> NETWORK DUMP <<<")

        for i, layer in enumerate(self.layout):
            net_type = "Feature Map" if hasattr(layer, "get_true_kernel_shape") else "Fully Populated"
            print(f"\n> Layer {i} ({net_type})")

            if net_type == "Feature Map":
                print(f"Input Shape (Internal): {layer.get_true_input_shape()}")
                print(f"Output Shape (Per Kernel): {layer.get_output_shape()}")

            print(f"Nodes In: {layer.get_total_nodes_in()}")
            print(f"Nodes Out: {layer.get_total_nodes_out()}")

            print("Data:")

            if net_type == "Feature Map":
                print("> Weights:")
                for weight in layer.weights:
                    print(f">> {weight.get_as_array()}")

            if net_type == "Fully Populated":
                print("> Weights:")
                print(f">> {layer.weights.get_as_array()}")

    @staticmethod
    def load(path):
        file = file_api.File(path)
        file.load()

        layer_codes = layers.LayerCodes()
        layout = []

        for layer_id in range(len(file.segments.keys())):
            layer_raw_data = file.segments[f"layer_{layer_id}"]

            code = layer_raw_data[:2]
            layer_data = layer_raw_data[2:]

            layer_class = layer_codes[code]
            layer_instance = layer_class.deserialize(layer_data)
            layout.append(layer_instance)

        return Network(tuple(layout))


if __name__ == "__main__":
    import viewer

    net = Network((
        layers.ConvolutedLayer((10, 10), (2, 2), filter_count=3, colour_depth=1, stride=2),
        layers.FullyConnectedLayer(48, 2, activations.ReLU)
    ))

    net.save("test.pyn")

    net = Network.load("test.pyn")

    print("made network")

    a, b = np.random.random((100, 100)), np.random.random((100, 100))

    training_data = [
        [a, np.array([1, 0])],
        [b, np.array([0, 1])],
    ]

    v = viewer.viewer()

    for i in range(50):
        net.train(training_data, training_data, 500, 0.005)
        print()

        for point in training_data:
            print(point, net.forward_pass(point[0]))

    input()
