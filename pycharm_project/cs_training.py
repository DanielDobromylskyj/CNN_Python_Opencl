from network import Network
import layers
import activations

import data_maker
import numpy as np
import os

OPENSLIDE_PATH = r'\openslide-win64\bin'
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(os.getcwd() + OPENSLIDE_PATH):
        import openslide
else:
    import openslide


class DocString:
    def __init__(self):
        """ This is a testing file for training misc networks """


trainingData = []



if __name__ == "__main__":
    import training_display_lite

    net = Network((
        layers.ConvolutedLayer((100, 100), (3, 3), filter_count=16, colour_depth=3, stride=1),
        layers.ConvolutedLayer((97 * 16 * 3, 97), (5, 5), filter_count=8, colour_depth=1, stride=3),
        layers.FullyConnectedLayer(384648, 1000, activations.Sigmoid),
        layers.FullyConnectedLayer(1000, 2, activations.Sigmoid),
    ))

    training_display_lite.Display.launch_threaded(net)

    net.save("start.pyn")
    #net = Network.load("training.pyn")

    l_rate = -1  # it so fucked the l_rate must be inverted ???

    net.train(trainingData[:5], trainingData[:5], 500, l_rate, show_stats=False)

    net.save("training.pyn")
