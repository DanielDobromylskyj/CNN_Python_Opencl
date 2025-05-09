from network import Network
import layers
import activations

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


with open("trainingPoints.txt", "r") as f:
    unprocessedTrainingData = eval(f.read())

trainingData = []
for segment in unprocessedTrainingData:
    slide = openslide.open_slide(segment["tiffPath"])
    for point in segment["points"]:
        img = slide.read_region(point[0], 0, (100, 100))
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)

        trainingData.append([
            arr / 255,
            [point[1][0], 0 if point[1][0] == 1 else 1]
        ])


if __name__ == "__main__":
    import training_display_lite

    net = Network((
        layers.ConvolutedLayer((100, 100), (5, 5), filter_count=15, colour_depth=3, stride=3),
        layers.FullyConnectedLayer(45600, 1000, activations.Sigmoid),
        layers.FullyConnectedLayer(1000, 2, activations.Sigmoid),
    ))

    training_display_lite.Display.launch_threaded(net)

    net.save("start.pyn")
    #net = Network.load("training.pyn")

    l_rate = -1

    net.train(trainingData[:5], trainingData[:5], 500, l_rate, show_stats=False)

    net.save("training.pyn")
