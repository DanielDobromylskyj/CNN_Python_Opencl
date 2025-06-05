from ..buffer import NetworkBuffer
import pyopencl as pycl

import math
import itertools
from functools import reduce
from operator import mul


class DefaultLayer:
    def __init__(self):
        self._cl = None
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

    def save(self, file):
        raise NotImplementedError("Class has not implemented serialize")

    @staticmethod
    def load(cl, file):
        raise NotImplementedError("Class has not implemented deserialize")

    @staticmethod
    def get_kernel_name():
        raise NotImplementedError("Class has not implemented get_kernel_name method, Unknown kernel required")

    def set_kernels(self, cl, kernels):
        self.__kernels = kernels
        self._cl = cl

    def execute_forward_kernel(self, function_name, shape, *args):
        if self.__kernels[0] is None:
            raise ValueError("No Forward kernel available / loaded.")

        print("Executing Forward kernel:", function_name, shape)

        kernel = getattr(self.__kernels[0], function_name)
        self.__execute_kernel(kernel, shape, *args)

    def execute_training_kernel(self, function_name, shape, *args):
        if self.__kernels[1] is None:
            raise ValueError("No Training kernel available / loaded.")

        print("Executing Training kernel:", function_name, shape)

        kernel = getattr(self.__kernels[1], function_name)
        self.__execute_kernel(kernel, shape, *args)

    def __execute_kernel(self, kernel, shape, *args):
        kernel(self._cl.queue, shape, None, *args)  # May need with batch executing with enqueue kernel range.
        self._cl.queue.flush()