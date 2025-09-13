from .buffer import EmptyNetworkBuffer, NetworkBuffer
import numpy as np


def format_with_batching(cl, inputs, batch) -> tuple[list | np.ndarray | NetworkBuffer, int]:
    if batch is not False:
        if type(inputs) in (list, tuple):
            return concatenate_batch_to_buffer(cl, inputs), len(inputs)

        elif batch:
            return inputs, len(inputs)

        else:
            return inputs, int(batch)

    return inputs, 1

def fill_empty_buffer_with_value(buffer: EmptyNetworkBuffer, value: float):
    size = np.prod(buffer.get_shape())
    array = np.full(size, value, dtype=buffer.get_dtype())
    buffer.write_to_buffer(array)

def concatenate_batch_to_buffer(cl, arrays):
    """ Warning: All arrays must be same size! """
    big_array = np.stack(arrays)  # if shape is consistent

    return NetworkBuffer(cl, big_array, big_array.shape)


class WeightInitCollection:
    def _get_shapes(self, input_shape, output_shape):
        if isinstance(input_shape, int):
            input_shape = [input_shape]
        if isinstance(output_shape, int):
            output_shape = [output_shape]
        return tuple(input_shape + output_shape)

    def ReLU(self, input_shape: list | tuple | int, output_shape: list | tuple | int) -> np.ndarray:
        shape = self._get_shapes(input_shape, output_shape)
        fan_in = np.prod(input_shape) if isinstance(input_shape, (list, tuple)) else input_shape
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * std

    def Sigmoid(self, input_shape: list | tuple | int, output_shape: list | tuple | int) -> np.ndarray:
        shape = self._get_shapes(input_shape, output_shape)
        fan_in = np.prod(input_shape) if isinstance(input_shape, (list, tuple)) else input_shape
        fan_out = np.prod(output_shape) if isinstance(output_shape, (list, tuple)) else output_shape
        std = np.sqrt(1.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * std


class BiasInitCollection:
    def _get_shape(self, output_shape):
        if isinstance(output_shape, int):
            return (output_shape,)
        elif isinstance(output_shape, (list, tuple)):
            return tuple(output_shape)
        else:
            raise ValueError("Invalid output_shape")

    def ReLU(self, input_shape: list | tuple | int, output_shape: list | tuple | int) -> np.ndarray:
        shape = self._get_shape(output_shape)
        return np.full(shape, 0.01)  # Small positive bias to help ReLU neurons fire early

    def Sigmoid(self, input_shape: list | tuple | int, output_shape: list | tuple | int) -> np.ndarray:
        shape = self._get_shape(output_shape)
        return np.zeros(shape)  # Neutral start for Sigmoid


def weight_init(input_shape: list | tuple | int, output_shape: list | tuple | int, activation: int):
    weight_inits = WeightInitCollection()

    match activation:
        case 1:  # ReLU
            return weight_inits.ReLU(input_shape, output_shape).astype(np.float32)

        case 2:  # Sigmoid
            return weight_inits.Sigmoid(input_shape, output_shape).astype(np.float32)

    return None


def bias_init(input_shape: list | tuple | int, output_shape: list | tuple | int, activation: int):
    bias_inits = BiasInitCollection()

    match activation:
        case 1:  # ReLU
            return bias_inits.ReLU(input_shape, output_shape).astype(np.float32)

        case 2:  # Sigmoid
            return bias_inits.Sigmoid(input_shape, output_shape).astype(np.float32)

    return None
