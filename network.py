import numpy as np

import buffers
import layers
import activations


# Todo - Implement saving layer data for backpropagation / gradient calculations

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
            raise InvalidNetwork(f"Layer {i+2} has a invalid number of inputs '{layout[0].get_total_nodes_in()}'")

        if previous_output_size <= 0:
            raise InvalidNetwork(f"Layer {i+1} has a invalid number of outputs '{previous_output_size}'")

        if previous_output_size != layer.get_total_nodes_in():
            raise InvalidNetwork(f"Layer {i+1} Outputs {previous_output_size} Values but Layer {i+2} Takes {layer.get_total_nodes_in()}")

        previous_output_size = layer.get_total_nodes_out()


class Network:
    def __init__(self, layout: tuple, validate_network=True):
        self.layout = layout

        if validate_network:
            validate_network_layout(self.layout)

    def forward_pass(self, inputs, save_layer_data=False, for_display=False):
        output_values = buffers.create_network_buffer_from_input(inputs)

        save_values = []

        for layer in self.layout:
            output_values = layer.forward(output_values)

            if for_display:
                save_values.append(output_values)

            if isinstance(output_values, buffers.NetworkBuffer) is False:  # likely a Filter output
                output_values = buffers.combine_buffers(output_values)

        if for_display:
            return save_values

        if isinstance(output_values, buffers.NetworkBuffer):
            return output_values.get_as_array()
        else:
            return [output_value.get_as_array() for output_value in output_values]

    def __calculate_gradients(self, inputs: np.ndarray, target: np.ndarray, learning_rate: float) -> list[tuple[buffers.Gradients, buffers.Gradients, buffers.Gradients]]:
        pass


if __name__ == "__main__":
    import viewer

    net = Network((
        layers.ConvolutedLayer((100, 100), (5, 5), filter_count=5, colour_depth=3),
        layers.FullyConnectedLayer(20*20*5, 2, activations.ReLU)
    ))
    print("made network")

    v = viewer.viewer()
    v.display(net, np.random.randn(30_000).astype(np.float32))
    #v.display(net, np.ones(30_000).astype(np.float32))
