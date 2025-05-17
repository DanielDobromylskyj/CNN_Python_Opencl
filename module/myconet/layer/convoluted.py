from .default import DefaultLayer
from .values import create_network_buffer
from ..buffers import Buffer, NetworkBuffer, Gradients


class ConvolutedLayer(DefaultLayer):
    def __init__(self, input_size: tuple[int, int], kernel_size: tuple[int, int], activation: int, filter_count: int,
                 colour_depth: int = 1, loading: bool = False, testing=None):  # todo - allow for sigmoid / other activations
        self.__filter_count = filter_count
        self.__input_size = input_size
        self.__kernel_size = kernel_size
        self.__colour_depth = colour_depth
        self.__activation = activation

        self.forward_core = load_core("kernel")

        self.weights = [
            NetworkBuffer(
                create_network_buffer((self.get_input_size(), self.get_output_size()), self.__activation, loading, testing),
                (self.get_weight_count(),)
            )
            for i in range(self.__filter_count)
        ] if loading is False else None

        self.biases = [
            NetworkBuffer(
                np.zeros((1,), dtype=np.float32) if not testing else np.full(self.get_bias_count(), testing[1], dtype=np.float32),
                (self.get_bias_count(),))
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

        return math.ceil(input_size[0] / kernel_size[0]), math.ceil(input_size[1] / kernel_size[1])

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
        ]) + f"..{self.__filter_count}..{self.__input_size}..{self.__kernel_size}..{self.__colour_depth}".encode()

    @staticmethod
    def deserialize(data):
        """ Turn a string of bytes into a layer instance"""
        weights, biases, filter_count, input_size, kernel_size, colour_depth = data.split(b"..")

        layer = ConvolutedLayer(
            eval(input_size.decode()),
            eval(kernel_size.decode()),
            int(filter_count.decode()),
            int(colour_depth.decode()),
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