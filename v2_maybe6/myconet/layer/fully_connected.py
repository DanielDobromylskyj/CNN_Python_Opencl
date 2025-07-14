import numpy as np

from .default import DefaultLayer
from .. import file_api
from .. import buffer
from ..buffer import NetworkBuffer
from ..util import concatenate_batch_to_buffer, weight_init, bias_init


class FullyConnected(DefaultLayer):
    def __init__(self, input_size: int, output_size: int, activation, is_loading=False):
        super().__init__()

        self.__input_size = input_size
        self.__output_size = output_size
        self.__activation = activation
        self.__is_loading = is_loading

        self.weights = None
        self.bias = None

    def init_values(self):
        if not self._cl:
            raise Exception("No OpenCL (cl) parsed before attempting to initialize")

        if self.__is_loading:
            return  # do not init

        # weight_init and bias_init take into account what activation and are in the util.py file

        self.weights = buffer.NetworkBuffer(
            self._cl,
            weight_init(self.__input_size, self.__output_size, self.__activation),
            (self.__input_size * self.__output_size, )
        )

        self.bias = buffer.NetworkBuffer(
            self._cl,
            bias_init(self.__input_size, self.__output_size, self.__activation),
            (self.__output_size,)
        )


    def get_node_count(self):
        return self.__input_size, self.__output_size


    def get_approximate_gpu_usage(self, data: np.ndarray | list, batch=False):
        single_output_usage = self.__output_size + self.__output_size * self.__input_size
        bytes_per_item = np.float32().nbytes

        if batch:
            return (single_output_usage + data[0].shape[0]) * bytes_per_item * len(data)
        else:
            return (single_output_usage + data.shape[0]) * bytes_per_item


    def forward(self, inputs: NetworkBuffer | list, wait=True, batch=False):
        batch_size = 1

        if batch:
            if type(inputs) in (list, tuple):
                batch_size = len(inputs)
                inputs = concatenate_batch_to_buffer(self._cl, inputs)

            else:
                batch_size = int(batch)

        outputs = buffer.create_empty_buffer(self._cl, self.__output_size * batch_size)
        unreduced_outputs = buffer.create_empty_buffer(self._cl, self.__output_size * self.__input_size * batch_size )

        event = self.execute_forward_kernel("forward",
                                             (self.__input_size, self.__output_size, batch_size),
                                             inputs.get_as_buffer(),
                                             unreduced_outputs.get_as_buffer(),
                                             self.weights.get_as_buffer(),
                                             np.int32(self.__input_size),
                                             np.int32(self.__output_size)
                                             )


        self.execute_forward_kernel("reduce_outputs",
                                             (self.__output_size, batch_size),
                                             unreduced_outputs.get_as_buffer(),
                                             outputs.get_as_buffer(),
                                             self.bias.get_as_buffer(),
                                             np.int32(self.__input_size),
                                             np.int32(self.__output_size),
                                             np.int32(self.__activation),
                                             wait_for=event
                                             ).wait()

        return outputs


    def forward_train(self, inputs: NetworkBuffer | list, batch=False):
        batch_size = 1

        if batch:
            if type(inputs) in (list, tuple):
                batch_size = len(inputs)
                inputs = concatenate_batch_to_buffer(self._cl, inputs)

            else:
                batch_size = int(batch)

        outputs = buffer.create_empty_buffer(self._cl, self.__output_size * batch_size)
        unactivated_outputs = buffer.create_empty_buffer(self._cl, self.__output_size * batch_size)
        unreduced_outputs = buffer.create_empty_buffer(self._cl, self.__output_size * self.__input_size * batch_size)

        event = self.execute_forward_kernel("forward",
                                    (self.__input_size, self.__output_size, batch_size),
                                    inputs.get_as_buffer(),
                                    unreduced_outputs.get_as_buffer(),
                                    self.weights.get_as_buffer(),
                                    np.int32(self.__input_size),
                                    np.int32(self.__output_size),
                                    )

        self.execute_training_kernel("reduce_outputs_forward",
                                    (self.__output_size, batch_size),
                                    unreduced_outputs.get_as_buffer(),
                                     outputs.get_as_buffer(),
                                     unactivated_outputs.get_as_buffer(),
                                     self.bias.get_as_buffer(),
                                     np.int32(self.__input_size),
                                     np.int32(self.__output_size),
                                     np.int32(self.__activation),
                                     wait_for=event
                                    ).wait()

        return outputs, unactivated_outputs, unreduced_outputs

    def backward(self, input_values: np.ndarray, error_gradients: NetworkBuffer, values: list, learning_rate: float):
        outputs, unactivated_outputs, unreduced_outputs = values

        layer_errors_unreduced = buffer.create_empty_buffer(self._cl, self.__input_size * self.__output_size)

        weight_gradients = buffer.create_empty_buffer(self._cl, self.__input_size * self.__output_size)
        bias_gradients = buffer.create_empty_buffer(self._cl, self.__output_size)

        self.log.true_debug(f"Unreduced Errors Buffer Size: {layer_errors_unreduced.get_shape()}")
        self.log.true_debug(f"Weight Buffer Size: {weight_gradients.get_shape()}")
        self.log.true_debug(f"Bias Buffer Size: {bias_gradients.get_shape()}")

        self.log.true_debug(f"Unreduced Outputs Buffer Size: {unreduced_outputs.get_shape()}")
        self.log.true_debug(f"Unactivated Outputs Buffer Size: {unactivated_outputs.get_shape()}")
        self.log.true_debug(f"Outputs Buffer Size: {outputs.get_shape()}")

        self.log.true_debug(f"Input Values Shape: {input_values.shape}")
        self.log.true_debug(f"Output Size: {self.__output_size}, Input Size: {self.__input_size}")
        self.log.true_debug(f"Activation: {self.__activation}, Learning Rate: {learning_rate}")

        self.log.true_debug(f"Kernel Shape: {(self.__input_size, self.__output_size)}")

        self.log.debug("Training backwards -> Backwards Kernel")
        self.execute_training_kernel(
            "backwards",
            (self.__input_size, self.__output_size),
            NetworkBuffer(self._cl, input_values, input_values.shape).get_as_buffer(),
            outputs.get_as_buffer(),
            unactivated_outputs.get_as_buffer(),
            self.weights.get_as_buffer(),
            error_gradients.get_as_buffer(),

            layer_errors_unreduced.get_as_buffer(),
            weight_gradients.get_as_buffer(),
            bias_gradients.get_as_buffer(),

            np.int32(self.__output_size),
            np.int32(self.__input_size),
            np.int32(self.__activation),
            np.float32(learning_rate)
        ).wait()

        self.log.debug("Training backwards -> Summing Errors")

        # Using CPU cos doing this on the GPU is a PAIN with float32 (or anything but ints)
        pre_summed = layer_errors_unreduced.get_as_array().reshape((self.__output_size, self.__input_size))
        summed = pre_summed.sum(axis=0).astype(np.float32) # Sum along output dimension -> result shape: (input_size, )
        layer_errors_reduced = buffer.NetworkBuffer(self._cl, summed, (self.__input_size,))

        return layer_errors_reduced, weight_gradients, bias_gradients

    def save(self, file, compress):
        file_api.encode_dict({
            "input_size" : self.__input_size,
            "output_size" : self.__output_size,
            "activation" : self.__activation,

            "weights": self.weights.get_as_array(),
            "biases" : self.bias.get_as_array(),
        }, file, compress)


    @staticmethod
    def load(cl, file, compressed):
        data = file_api.decode_dict(file, compressed)
        layer = FullyConnected(input_size=data["input_size"], output_size=data["output_size"], activation=data["activation"], is_loading=True)
        layer.weights = buffer.NetworkBuffer(cl, data["weights"], (data["input_size"], data["output_size"]))
        layer.bias = buffer.NetworkBuffer(cl, data["biases"], (data["output_size"],))

        return layer


    @staticmethod
    def get_kernel_name():
        return "full_pop"

    def __str__(self):
        return f"FullyConnected(input={self.__input_size}, output={self.__output_size}, activation={self.__activation})"

    def release(self):
        self.weights.release()
        self.bias.release()

