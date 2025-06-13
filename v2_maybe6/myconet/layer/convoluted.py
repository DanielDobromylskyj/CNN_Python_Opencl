from ..buffer import NetworkBuffer
import pyopencl as pycl


def ensure_input_is_3D(shape):
    if len(shape) == 3:
        return shape

    if len(shape) == 2:
        return shape[0], shape[1], 1

    raise InputError("Cannot Make Convoluted Layer with input shape that is not 2D or 3D")

def ensure_kernel_is_2D(shape):
    if len(shape) == 2:
        return shape

    raise InputError("Cannot Make Convoluted Layer with kernel shape that is not 2D")

def calculate_output_shape(input_shape, kernel_shape, stride):
    input_width, input_height, channels = input_shape
    kernel_width, kernel_height = kernel_shape

    output_width = ((input_width - kernel_width) // stride) + 1
    output_height = ((input_height - kernel_height) // stride) + 1

    return channels, output_width, output_height


def ensure_stride_is_int(stride):
    if type(stride) is int:
        return stride

    raise InputError("Cannot Make Convoluted Layer with stride that is not int")


class Convoluted:
    def __init__(self, input_shape: tuple[int, int, int] | tuple[int, int], kernel_shape: tuple[int, int], stride:int, activation:int, is_loading=False):
        self.__input_shape = ensure_input_is_3D(input_shape)
        self.__kernel_shape = ensure_kernel_is_2D(kernel_shape)
        self.__stride = ensure_stride_is_int(stride)

        self.__output_shape = calculate_output_shape(self.__input_shape, self.__kernel_shape, self.__stride)
        self.__activation = activation
        self.__is_loading = is_loading

        self.weights = None
        self.bias = None

    def init_values(self):
        if not self._cl:
            raise Exception("No OpenCL (cl) parsed before attempting to initialize")

        if self.__is_loading:
            return  # do not init

        self.weights = [buffer.NetworkBuffer(
            self._cl, np.ones((self.__kernel_shape[0] * self.__kernel_shape[1]), dtype=np.float32),  # todo
            (self.__input_size * self.__output_size,))

            for _ in range(self.____input_shape[2])  # Channels (Like RGB), We want a different weight set for each.
            ]

        self.bias = buffer.NetworkBuffer(
            self._cl, np.zeros((self.__output_size[1] * self.__output_shape[2]), dtype=np.float32),  # todo
            (self.__output_size,)
        )

    def forward(self, inputs: NetworkBuffer):
        raise NotImplementedError("Class has not implemented forward method")

    def forward_train(self, inputs: NetworkBuffer):
        raise NotImplementedError("Class has not implemented forward (Training) method")

    def backward(self, input_values: NetworkBuffer, error_gradients: NetworkBuffer, values: list, learning_rate: float):
        raise NotImplementedError("Class has not implemented backward method")

    def get_node_count(self):
        return self.__input_shape[0] * self.__input_shape[1] * self.__input_shape[2], self.__output_shape[0] * self.__output_shape[1]

    def save(self, file, compress):
        raise NotImplementedError("Class has not implemented serialize")

    @staticmethod
    def load(cl, file, compressed):
        raise NotImplementedError("Class has not implemented deserialize")

    @staticmethod
    def get_kernel_name():
        return "kernel"

    def release(self):
        raise NotImplementedError("Class has not implemented release")

    def __str__(self):
        return f"{self.__class__.__name__}(string_method=NotImplemented)"

