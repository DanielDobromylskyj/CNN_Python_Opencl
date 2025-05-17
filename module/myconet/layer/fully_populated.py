from .default import DefaultLayer
from .values import create_network_buffer
from ..buffers import Buffer, NetworkBuffer, Gradients


class FullyConnectedLayer(DefaultLayer):
    def __init__(self, input_size: int, output_size: int, activation: int, loading: bool = False, testing=None, ADAM_data=None):
        self.__input_size = input_size
        self.__output_size = output_size
        self.__activation = activation

        self.weights = NetworkBuffer(
                create_network_buffer((self.get_input_size(), self.get_output_size()), self.__activation, loading, testing),
                (self.get_weight_count(),)
            )

        self.biases = NetworkBuffer(
            np.zeros(self.get_bias_count()) if not testing else np.full(self.get_bias_count(), testing[1], dtype=np.float32),
            (self.get_bias_count(),)
        ) if loading is False else None

        if ADAM_data:
            beta1, beta2 = ADAM_data
            self.optimisers_weights = ADAM(self.get_weight_count(), beta1, beta2)
            self.optimisers_biases = ADAM(self.get_bias_count(), beta1, beta2)
        else:
            self.optimisers_weights = None
            self.optimisers_biases = None

        self.forward_core = load_core("full_pop")

    def get_input_size(self):
        """ Returns the size of inputs """
        return self.__input_size

    def get_output_size(self):
        """ Returns the size of outputs """
        return self.__output_size

    def get_weight_count(self):
        """ Returns the number of weights """
        return self.__input_size * self.__output_size

    def get_bias_count(self):
        """ Returns the number of biases """
        return self.__output_size

    def get_total_nodes_in(self) -> int:
        """ Returns the total number of nodes in"""
        return self.get_input_size()

    def get_total_nodes_out(self) -> int:
        """ Returns the total number of nodes out"""
        return self.get_output_size()

    def forward(self, inputs: NetworkBuffer, save_layer_data=False) -> NetworkBuffer | tuple[
        NetworkBuffer, NetworkBuffer]:
        """ Performs a forward pass over all the input data """

        output = NetworkBuffer(np.zeros(self.get_output_size(), dtype=np.float32), self.get_output_size())
        unreduced_outputs = NetworkBuffer(np.empty(self.get_weight_count(), dtype=np.float32), self.get_weight_count())
        unactivated_outputs = NetworkBuffer(np.empty(self.get_output_size(), dtype=np.float32), self.get_output_size())

        self.forward_core.forward(queue, (self.get_input_size(), self.get_output_size()), None,
                                  inputs.get_as_buffer(), unreduced_outputs.get_as_buffer(),
                                  self.weights.get_as_buffer(), np.int32(self.get_input_size())
                                  ).wait()

        self.forward_core.reduce_outputs(queue, (self.get_output_size(),), None,
                                         unreduced_outputs.get_as_buffer(), output.get_as_buffer(),
                                         unactivated_outputs.get_as_buffer(), self.biases.get_as_buffer(),
                                         np.int32(self.get_input_size()), np.int32(self.get_output_size()),
                                         np.int32(self.__activation)
                                         ).wait()

        if save_layer_data:
            return output, unactivated_outputs

        return output

    def backward(self, inputs: NetworkBuffer, outputs_activated: NetworkBuffer,
                 outputs_unactivated: NetworkBuffer, output_error_gradients: Gradients,
                 learning_rate: float) -> tuple[Gradients, Gradients, Gradients]:
        """ Calculates gradients and errors using backpropagation"""
        core = load_core("training/full_pop")

        # Create Gradients
        weight_gradients = Gradients(np.zeros(self.get_weight_count(), dtype=np.float32))
        bias_gradients_unreduced = Gradients(np.zeros(self.get_weight_count(), dtype=np.float32))
        bias_gradients = Gradients(np.zeros(self.get_bias_count(), dtype=np.float32))
        input_gradients_unreduced = Gradients(np.zeros(self.get_weight_count(), dtype=np.float32))
        input_gradients = Gradients(np.zeros(self.get_input_size(), dtype=np.float32))

        # Step 1/3 - Perform calculations
        core.backwards(queue, (self.get_input_size(), self.get_output_size()), None,
                       inputs.get_as_buffer(), outputs_activated.get_as_buffer(),
                       outputs_unactivated.get_as_buffer(), self.weights.get_as_buffer(),
                       self.biases.get_as_buffer(), output_error_gradients.get_as_buffer(),
                       input_gradients_unreduced.get_as_buffer(), weight_gradients.get_as_buffer(),
                       bias_gradients_unreduced.get_as_buffer(), np.int32(self.get_input_size()),
                       np.int32(self.get_output_size()), np.int32(self.__activation), np.float32(learning_rate)
                       ).wait()

        # Step 2/3 - Reduce Input Gradients
        core.reduce_input_error_gradients(queue, (self.get_input_size(),), None,
                                          input_gradients_unreduced.get_as_buffer(),
                                          input_gradients.get_as_buffer(),
                                          np.int32(self.get_input_size()),
                                          np.int32(self.get_output_size())
                                          ).wait()

        # Step 3/3 - Reduce Bias Gradients
        core.reduce_bias_gradients(queue, (self.get_output_size(),), None,
                                   bias_gradients_unreduced.get_as_buffer(),
                                   bias_gradients.get_as_buffer(),
                                   np.int32(self.get_input_size()),
                                   np.int32(self.get_output_size())
                                   ).wait()

        if np.any(np.isnan(weight_gradients.get_as_array()) | np.isinf(
                weight_gradients.get_as_array())):
            raise ValueError("[FULL POP] NaN or Inf Found In Weights (NaN, Inf): " + str(
                np.any(np.isnan(weight_gradients.get_as_array()))) + ", " + str(
                np.any(np.isinf(weight_gradients.get_as_array()))))

        return input_gradients, weight_gradients, bias_gradients

    def apply_gradients(self, weight_gradients: Gradients, bias_gradients: Gradients, count: int) -> None:
        """ Applies gradients to weights and biases, using adam if enabled """
        if not self.optimisers_weights:
            self.weights += weight_gradients / count
            self.biases += bias_gradients / count

        else:
            weights = self.weights.get_as_array()
            biases = self.biases.get_as_array()

            self.optimisers_weights.step()
            adjusted_weights = self.optimisers_weights.optimise(weights, weight_gradients)
            self.weights = NetworkBuffer(adjusted_weights, adjusted_weights.shape)

            self.optimisers_biases.step()
            adjusted_biases = self.optimisers_biases.optimise(biases, bias_gradients)
            self.biases = NetworkBuffer(adjusted_biases, adjusted_biases.shape)

    def serialize(self):
        """ Returns a serialized version of the layers in bytes"""
        return base64.b64encode(self.weights.get_as_array().tobytes()) + b".." + base64.b64encode(
            self.biases.get_as_array().tobytes()) + f"..{self.__input_size}..{self.__output_size}..{self.__activation}".encode()

    @staticmethod
    def deserialize(data):
        """ Converts bytes into a layer instance"""
        weights, biases, input_size, output_size, activation = data.split(b"..")

        layer = FullyConnectedLayer(
            eval(input_size.decode()),
            eval(output_size.decode()),
            int(activation.decode()),
            loading=True
        )

        layer.weights = create_network_buffer_from_input(
            np.frombuffer(base64.b64decode(weights), dtype=np.float32)
        )

        layer.biases = create_network_buffer_from_input(
            np.frombuffer(base64.b64decode(biases), dtype=np.float32)
        )

        return layer
