import numpy as np

from .default import DefaultLayer
from .. import buffer
from .. import file_api


class FullyConnected(DefaultLayer):
    def __init__(self, input_size, output_size, activation, is_loading=False):
        super(FullyConnected, self).__init__()

        self.__input_size = input_size
        self.__output_size = output_size
        self.__activation = activation

        self.weights = buffer.NetworkBuffer(
            np.ones((input_size, output_size), dtype=np.float32),  # todo
            (input_size, output_size)
        ) if not is_loading else None

        self.bias = buffer.NetworkBuffer(
            np.zeros((output_size,), dtype=np.float32),  # todo
            (output_size,)
        ) if not is_loading else None


    def save(self, file):
        file_api.encode_dict({
            "input_size" : self.__input_size,
            "output_size" : self.__output_size,
            "activation" : self.__activation,

            "weights": self.weights.get_as_array(),
            "biases" : self.bias.get_as_array(),
        }, file)


    @staticmethod
    def load(file):
        data = file_api.decode_dict(file)
        layer = FullyConnected(input_size=data["input_size"], output_size=data["output_size"], activation=data["activation"], is_loading=True)
        layer.weights = buffer.NetworkBuffer(data["weights"], (data["input_size"], data["output_size"]))
        layer.bias = buffer.NetworkBuffer(data["biases"], (data["output_size"],))

        return layer


