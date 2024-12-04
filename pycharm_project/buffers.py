import pyopencl as cl
import numpy as np


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


def create_network_buffer_from_input(values):
    np_array = np.array(values)
    return NetworkBuffer(np_array.ravel().astype(np.float32), np_array.shape)


def combine_buffers(buffers):  # todo - find a better / faster way of doing this
    summed = np.array([], dtype=np.float32)
    for buffer in buffers:
        summed = np.append(summed, buffer.get_as_array())
    return NetworkBuffer(summed, summed.shape)


def convert_gradients_to_buffer_list(buffer, chunk_size):
    data = buffer.get_as_array()
    chunks = [Gradients(data[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]
    return BufferList(chunks)


def rearrange_feature_map_output(outputs):
    input_gradients = []
    weight_gradients = []
    bias_gradients = []

    for kernel in outputs:
        input_gradients.append(kernel[0])
        weight_gradients.append(kernel[1])
        bias_gradients.append(kernel[2])

    return BufferList(input_gradients), BufferList(weight_gradients), BufferList(bias_gradients)


class NetworkBuffer:
    def __init__(self, data: np.ndarray, shape: tuple[int]):
        self.__buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.USE_HOST_PTR, data.nbytes, hostbuf=data)
        self.__shape = shape

    def get_shape(self):
        return self.__shape

    def is_image(self):
        return len(self.get_shape()) > 1

    def is_rgb_image(self):
        return len(self.get_shape()) > 2

    def get_as_buffer(self):
        return self.__buffer

    def get_as_array(self):
        output = np.empty(self.get_shape(), dtype=np.float32)
        cl.enqueue_copy(queue, output, self.get_as_buffer()).wait()
        return output

    def __add__(self, other):
        if isinstance(other, NetworkBuffer) or isinstance(other, Gradients):
            return NetworkBuffer(self.get_as_array() + other.get_as_array(), self.__shape)
        return NetworkBuffer(self.get_as_array() + other, self.__shape)

    def __truediv__(self, other):
        return NetworkBuffer(self.get_as_array() / other, self.__shape)


class BufferList:
    def __init__(self, buffers):
        self.__buffers = buffers

    def get_as_buffer(self, index=None):
        if index is None:
            return combine_buffers(self.__buffers).get_as_buffer()
        else:
            return self.__buffers[index].get_as_buffer()

    def get_network_buffer(self, index):
        return self.__buffers[index]

    def get_as_array(self, index=None):
        if index is None:
            return tuple([
                buffer.get_as_array()
                for buffer in self.__buffers
            ])

        else:
            return self.__buffers[index].get_as_array()

    def __len__(self):
        return len(self.__buffers)

    def __iter__(self):
        for buffer in self.__buffers:
            yield buffer

    def __getitem__(self, item):
        return self.__buffers[item]

    def __setitem__(self, key, value):
        self.__buffers[key] = value


class Gradients:
    def __init__(self, gradients: np.ndarray):
        self.__gradients = cl.Buffer(ctx, mf.READ_WRITE | mf.USE_HOST_PTR, gradients.nbytes, hostbuf=gradients)
        self.__shape = gradients.shape

    def get_shape(self):
        return self.__shape

    def get_as_buffer(self):
        return self.__gradients

    def get_as_array(self):
        output = np.empty(self.get_shape(), dtype=np.float32)
        cl.enqueue_copy(queue, output, self.get_as_buffer()).wait()
        return output

    def __add__(self, other_gradients):
        return Gradients(self.get_as_array() + other_gradients.get_as_array())

    def __truediv__(self, other):
        return Gradients(self.get_as_array() / other)

    def __neg__(self):
        return Gradients(self.get_as_array() * -1)
