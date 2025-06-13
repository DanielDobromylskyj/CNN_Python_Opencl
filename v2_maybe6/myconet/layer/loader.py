from .fully_connected import FullyConnected
from .convoluted import Convoluted


lookup = {
    FullyConnected: 1,
    Convoluted: 2,
}

def layer_to_code(layer):
    if layer.__class__ in lookup:
        return lookup[layer.__class__]

    raise NotImplementedError(f"Unknown layer type: {layer.__class__.__name__}")


def code_to_layer(code):
    for key, value in lookup.items():
        if code == value:
            return key

    raise NotImplementedError(f"Unknown layer code: {code}")