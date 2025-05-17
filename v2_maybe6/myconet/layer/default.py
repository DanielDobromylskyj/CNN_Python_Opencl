from ..buffer import NetworkBuffer


class DefaultLayer:
    def __init__(self):
        pass

    def forward(self, input: NetworkBuffer):
        raise NotImplementedError("Class has not implemented forward method")

    def backward(self, input: NetworkBuffer):
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
