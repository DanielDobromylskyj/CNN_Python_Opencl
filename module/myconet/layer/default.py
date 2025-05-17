from ..buffers import Buffer, NetworkBuffer, Gradients


class LayerCodes:
    def __init__(self):
        """ Shorthand codes for layer types (Used for storage)"""
        self.__code_to_layer = {
            "FP": FullyConnectedLayer,
            "CV": ConvolutedLayer
        }

        self.__layer_to_code = {v: k for k, v in self.__code_to_layer.items()}

    def __getitem__(self, item):
        if type(item) is bytes:
            return self.__code_to_layer[item.decode()]
        return self.__layer_to_code[item.__class__].encode()


class DefaultLayer:
    def forward(self, inputs: NetworkBuffer, save_layer_data=False):
        """ Perform a forward pass over the layer """
        raise NotImplementedError

    def backward(self, inputs: NetworkBuffer, outputs_activated: NetworkBuffer, outputs_unactivated: NetworkBuffer,
                 output_error_gradients: Gradients,
                 learning_rate: float) -> tuple[Gradients, Gradients, Gradients]:
        """ Calculate the gradients and errors for weights, biases and the 'next' layer """
        raise NotImplementedError

    def apply_gradients(self, weight_gradients: Gradients | BufferList, bias_gradients: Gradients | BufferList,
                        count: int) -> None:
        """ Applies the gradients we have calculated """
        raise NotImplementedError

    def get_total_nodes_in(self):
        raise NotImplementedError("Layer does not have a implemented 'get_total_nodes_in' method")

    def get_total_nodes_out(self):
        raise NotImplementedError("Layer does not have a implemented 'get_total_nodes_out' method")

    def serialize(self):
        raise NotImplementedError

    @staticmethod
    def deserialize(data):
        raise NotImplementedError