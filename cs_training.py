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
    net = Network((
        layers.ConvolutedLayer((100, 100), (5, 5), filter_count=3, colour_depth=3),
        layers.FullyConnectedLayer(400 * 3, 2, activations.ReLU)
    ))

    l_rate = 0.1

    net.train([trainingData[0]], [trainingData[0]], 500, l_rate)

    net.save("training.pyn")
