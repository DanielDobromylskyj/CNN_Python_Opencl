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

    def add(self, other_gradients):
        return Gradients(self.get_as_array() + other_gradients.get_as_array())

    def divide(self, value: int):
        return Gradients(self.get_as_array() / value)
