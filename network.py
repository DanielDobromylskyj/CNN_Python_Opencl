import numpy as np

import buffers
import layers
import activations


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
                     for_display=False):  # todo - save_layer_data -> return both activated and unactivated / None if data is not activated
        output_values = buffers.create_network_buffer_from_input(inputs)

        save_values = [] if not save_layer_data else [(output_values, None)]

        for layer in self.layout:
            output_values = layer.forward(output_values, save_layer_data)

            if for_display:
                save_values.append(output_values)

            if save_layer_data:  # this is useless right? (unpacking them)
                output_values, other_value = output_values
                save_values.append((output_values, other_value))

            if isinstance(output_values, buffers.NetworkBuffer) is False:  # likely a Filter output
                output_values = buffers.combine_buffers(output_values)

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

        if isinstance(data, list) or isinstance(data, tuple):
            if isinstance(data[0], buffers.NetworkBuffer):
                return buffers.combine_buffers(data)

            return buffers.NetworkBuffer(np.array(data, dtype=np.float32), (len(data),))

        if isinstance(data, np.ndarray):
            return buffers.NetworkBuffer(data, data.shape)

        raise ValueError(f"Could not ensure data is a network buffer or convert to one\nData:\n{data}")

    def backward_pass(self, inputs: np.ndarray, target: np.ndarray, learning_rate: float):
        data = self.forward_pass(inputs, save_layer_data=True)
        outputs = data[-1][0].get_as_array()

        error = target - outputs

        output_error_gradients = self.__convert_to_gradients(error)

        for layer_index in range(len(self.layout) - 1, -1, -1):
            print("Backprop On Layer Index:", layer_index)

            # Extract Inputs
            activated_inputs, unactivated_inputs = data[layer_index]

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


if __name__ == "__main__":
    import viewer

    net = Network((
        layers.ConvolutedLayer((100, 100), (5, 5), filter_count=5, colour_depth=3),
        layers.FullyConnectedLayer(20 * 20 * 5, 2, activations.ReLU)
    ))
    print("made network")

    rand_data = np.random.randn(30_000).astype(np.float32)

    l_rate = 0.1
    net.backward_pass(rand_data, np.array([0, 1]), l_rate)
    net.backward_pass(rand_data, np.array([0, 1]), l_rate)
    net.backward_pass(rand_data, np.array([0, 1]), l_rate)

    v = viewer.viewer()
    v.display(net, np.random.randn(30_000).astype(np.float32))
    # v.display(net, np.ones(30_000).astype(np.float32))
