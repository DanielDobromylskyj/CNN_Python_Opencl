from ..buffer import NetworkBuffer
import pyopencl as cl


class DefaultLayer:
    def __init__(self):
        self.__queue = None
        self.__kernels: tuple[None | cl.Program, ...] = (None, None)

    def forward(self, inputs: NetworkBuffer):
        raise NotImplementedError("Class has not implemented forward method")

    def backward(self, inputs: NetworkBuffer):
        raise NotImplementedError("Class has not implemented backward method")

    def get_node_count(self):
        raise NotImplementedError("Class has not implemented get_node_count method")

    def save(self, file):
        raise NotImplementedError("Class has not implemented serialize")

    @staticmethod
    def load(file):
        raise NotImplementedError("Class has not implemented deserialize")

    @staticmethod
    def get_kernel_name():
        raise NotImplementedError("Class has not implemented get_kernel_name method, Unknown kernel required")

    def set_kernels(self, kernels):
        self.__kernels = kernels

    def set_queue(self, queue):
        self.__queue = queue

    def execute_forward_kernel(self, function_name, shape, *args):
        if self.__kernels[0] is None:
            raise ValueError("No Forward kernel available / loaded.")

        kernel = getattr(self.__kernels[0], function_name)
        kernel(self.__queue, shape, None, *args)

    def execute_training_kernel(self, function_name, shape, *args):
        if self.__kernels[1] is None:
            raise ValueError("No Training kernel available / loaded.")

        kernel = getattr(self.__kernels[1], function_name)
        kernel(self.__queue, shape, None, *args)
