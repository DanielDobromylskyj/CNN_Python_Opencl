from . import file_api
from .layer import loader

class InvalidNetwork(Exception):
    pass


class Network:
    def __init__(self, layout: tuple, verify=True):
        self.layout = layout

        if verify:
            self.validate_layout()

    def validate_layout(self):
        for i in range(len(self.layout) - 1):
            nodes_out = self.layout[i].get_node_count()[0]
            nodes_in = self.layout[i + 1].get_node_count()[1]
            if nodes_in != nodes_out:
                raise InvalidNetwork(
                    f"Layout Invalid -> Layer {i+1} outputs {nodes_out}, Yet Layer {i+2} takes {nodes_in} inputs."
                )


    def capture_forward(self, inputs):  # Use for training, returns extra data
        return  # todo

    def forward(self, inputs):  # Not for training. Optimised for speed, use capture_forward(input)
        input_buffer: buffer.NetworkBuffer = buffer.create_network_buffer_from_input(inputs)

        for layer in self.layout:
            input_buffer = layer.forward(input_buffer)
        
        return input_buffer.get_as_array


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
        layout = []
        with open(path, 'rb') as f:
            layer_count = file_api.decode_int(f)

            layout = tuple([
                loader.code_to_layer(file_api.decode_int(f)).load(f)
                for i in range(layer_count)
            ])

        return Network(layout)



