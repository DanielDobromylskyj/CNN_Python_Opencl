from .. import buffer

class Optimiser:
    def __init__(self, network):
        self.net = network

    def apply_gradients(self, gradients):
        cl = self.net.cl

        for layer_index in range(len(self.net.layout)):
            weight_grads, bias_grads = gradients[len(self.net.layout) - layer_index - 1]

            weights = self.net.layout[layer_index].weights.get_as_array()
            biases = self.net.layout[layer_index].bias.get_as_array()

            weights += weight_grads
            biases += bias_grads

            self.net.layout[layer_index].weights = buffer.NetworkBuffer(cl, weights, weights.shape)
            self.net.layout[layer_index].bias = buffer.NetworkBuffer(cl, biases, biases.shape)






