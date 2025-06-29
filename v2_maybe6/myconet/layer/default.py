from ..buffer import NetworkBuffer
import pyopencl as pycl

import math
import itertools
from functools import reduce
from operator import mul


def chunk_ranges(global_size, max_sizes):
    chunks = []
    for i, size in enumerate(global_size):
        max_size = max_sizes[i] if i < len(max_sizes) else max_sizes[-1]

        chunk_list = []
        start = 0
        while start < size:
            chunk_list.append((start, min(max_size, size - start)))
            start += max_size
        chunks.append(chunk_list)
    return chunks




class DefaultLayer:
    def __init__(self):
        self._cl = None
        self.__cl_kernel_old = None
        self.log = None
        self.__kernels: tuple[None | pycl.Program, ...] = (None, None)

    def init_values(self):
        raise NotImplementedError("Class has no init_values function")

    def forward(self, inputs: NetworkBuffer):
        raise NotImplementedError("Class has not implemented forward method")

    def forward_train(self, inputs: NetworkBuffer):
        raise NotImplementedError("Class has not implemented forward (Training) method")

    def backward(self, input_values: NetworkBuffer, error_gradients: NetworkBuffer, values: list, learning_rate: float):
        raise NotImplementedError("Class has not implemented backward method")

    def get_node_count(self):
        raise NotImplementedError("Class has not implemented get_node_count method")

    def save(self, file, compress):
        raise NotImplementedError("Class has not implemented serialize")

    @staticmethod
    def load(cl, file, compressed):
        raise NotImplementedError("Class has not implemented deserialize")

    @staticmethod
    def get_kernel_name():
        raise NotImplementedError("Class has not implemented get_kernel_name method, Unknown kernel required")

    def set_logger(self, logger):
        self.log = logger

    def set_kernels(self, cl, kernels):
        self.__kernels = kernels
        self._cl = cl

    def change_cl(self, cl):
        self.__cl_kernel_old = self._cl
        self._cl = cl

    def restore_cl(self):
        self._cl = self.__cl_kernel_old

    def execute_forward_kernel(self, function_name, shape, *args):
        if self.__kernels[0] is None:
            raise ValueError("No Forward kernel available / loaded.")

        self.log.debug("Executing Forward kernel:", function_name, shape)

        kernel = getattr(self.__kernels[0], function_name)
        return self.__execute_kernel(kernel, shape, *args)

    def execute_training_kernel(self, function_name, shape, *args):
        if self.__kernels[1] is None:
            raise ValueError("No Training kernel available / loaded.")

        self.log.debug("Executing Training kernel:", function_name, shape)

        kernel = getattr(self.__kernels[1], function_name)
        return self.__execute_kernel(kernel, shape, *args)

    def __execute_kernel(self, kernel, shape, *args):
        return kernel(self._cl.queue, shape, None, *args)

    def release(self):
        raise NotImplementedError("Class has not implemented release")

    def await_kernels(self):
        self._cl.queue.finish()

    def __str__(self):
        return f"{self.__class__.__name__}(string_method=NotImplemented)"
