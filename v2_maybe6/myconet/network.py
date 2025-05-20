import pyopencl as cl

from .core.load import load_kernel, load_training_kernel
from . import file_api, buffer
from .layer import loader


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


    def __ready_kernels(self, load_training_kernels=False):
        self.__kernels = {}

        for layer in self.layout:
            if layer.__class__.__name__ not in self.__kernels:
                self.__kernels[layer.__class__.__name__] = (
                    load_kernel(self.cl, layer.get_kernel_name()),
                    load_training_kernel(self.cl, layer.get_kernel_name()) if load_training_kernels else None
                )

            layer.set_kernels(self.cl, self.__kernels[layer.__class__.__name__])
            layer.init_values()

    def validate_layout(self):
        for i in range(len(self.layout) - 1):
            nodes_out = self.layout[i].get_node_count()[1]
            nodes_in = self.layout[i + 1].get_node_count()[0]
            if nodes_in != nodes_out:
                raise InvalidNetwork(
                    f"Layout Invalid -> Layer {i+1} outputs {nodes_out}, Yet Layer {i+2} takes {nodes_in} inputs."
                )


    def capture_forward(self, inputs):  # Use for training, returns extra data
        input_buffer: buffer.NetworkBuffer = buffer.create_network_buffer_from_input(self.cl, inputs)

        extra_data = []

        for layer in self.layout:
            values = layer.forward_train(input_buffer)

            extra_data.append([value.get_as_array() for value in values])
            input_buffer = values[0]

        return input_buffer.get_as_array(), extra_data


    def forward(self, inputs):  # Not for training. Optimised for speed, use capture_forward(input)
        input_buffer: buffer.NetworkBuffer = buffer.create_network_buffer_from_input(self.cl, inputs)

        for layer in self.layout:
            input_buffer = layer.forward(input_buffer)
        
        return input_buffer.get_as_array()

    def backward(self, inputs: buffer.NetworkBuffer, target: np.ndarray | np.array):
        self.capture_forward(inputs)

    def train(self, training_data, learning_rate):
        self.__ready_kernels(load_training_kernels=True)
        # todo


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
