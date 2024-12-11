import os
import time
import numpy as np

import network
import layers
import activations
import buffers

OPENSLIDE_PATH = r'\openslide-win64\bin'
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(os.getcwd() + OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def test_performance(net):
    slide = openslide.open_slide("trainingData/16772.tiff")

    region_read_time = 0
    network_time = 0

    test_points = 200
    for i in range(test_points):
        start = time.time()
        region = slide.read_region((1500, 1500), 0, (100, 100)).convert("RGB")
        region_array = np.asarray(region, dtype=np.float32) / 255
        region_read_time += time.time() - start

        start = time.time()
        net.forward_pass(region_array)
        network_time += time.time() - start

    region_read_time = round(region_read_time / test_points * 1000)
    network_time = round(network_time / test_points * 1000)

    total_time = network_time + region_read_time

    print("> PERFORMANCE TEST <")
    print(f"Status (20ms): {'PASSED' if total_time <= 20 else 'FAILED'}")
    print(f"Status (50ms): {'PASSED' if total_time <= 50 else 'FAILED'}")
    print(f"Total: {total_time}ms, Load: {region_read_time}ms, Network: {network_time}ms")
    print("\n")


def test_accuracy(net):
    with open("trainingPoints.txt", "r") as f:
        unprocessedTrainingData = eval(f.read())

    tests = 0
    wrong = 0
    false_positive = 0
    false_negative = 0
    inconclusive = 0


    for segment in unprocessedTrainingData:
        slide = openslide.open_slide(segment["tiffPath"])
        for point in segment["points"]:
            img = slide.read_region(point[0], 0, (100, 100))
            img = img.convert("RGB")
            image_data_normalised = np.asarray(img, dtype=np.float32) / 255

            target = [point[1][0], 0 if point[1][0] == 1 else 1]

            output = net.forward_pass(image_data_normalised)
            tests += 1

            if output[0] > 0.5 and output[1] > 0.5:
                inconclusive += 1
                wrong += 1
            elif output[0] < 0.5 and output[1] < 0.5:
                inconclusive += 1
                wrong += 1

            elif target[0] == 1:
                if output[0] > 0.5 and output[1] < 0.5:
                    pass  # correct
                else:
                    false_negative += 1
                    wrong += 1

            elif target[1] == 1:
                if output[0] < 0.5 and output[1] > 0.5:
                    pass  # correct
                else:
                    false_positive += 1
                    wrong += 1

    print("> ACCURACY TEST <")
    print(f"Tested {tests} cases")
    print(f"Accuracy: {round(100 * (1 - (wrong / tests)))}%")
    print(f"False Negative Rate: {round(100 * ((false_negative / tests)))}%")
    print(f"False Positive Rate: {round(100 * ((false_positive / tests)))}%")
    print(f"Inconclusive Rate: {round(100 * ((inconclusive / tests)))}%")
    print("\n")


def test_output_validity_basic():
    print("> VALIDATION TEST BASIC <")

    # TEST NETWORK 1
    test_number = 1
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_1 = network.Network((
        layers.FullyConnectedLayer(5, 3, activations.Sigmoid,
                                   testing=[1, 0]),

        layers.FullyConnectedLayer(3, 2, activations.Sigmoid,
                                   testing=[1, 0]),
    ))

    outputs = test_net_1.forward_pass([1, 2, 3, 4, 5])
    expected = np.array([0.952574, 0.952574], dtype=np.float32)
    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")

    # TEST NETWORK 2
    test_number = 2
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_2 = network.Network((
        layers.FullyConnectedLayer(1, 2, activations.Sigmoid,
                                   testing=[1, 0]),

        layers.FullyConnectedLayer(2, 1, activations.ReLU,
                                   testing=[1, 0]),
    ))

    outputs = test_net_2.forward_pass([1])
    expected = np.array([1.4621172], dtype=np.float32)
    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")

    # TEST NETWORK 3
    test_number = 3
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_3 = network.Network((
        layers.FullyConnectedLayer(1, 2, activations.Sigmoid,
                                   testing=[1, 0]),

        layers.FullyConnectedLayer(2, 1, activations.ReLU,
                                   testing=[1, 0]),
    ))

    outputs = test_net_3.forward_pass([5.3])
    expected = np.array([1.9900664], dtype=np.float32)
    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")


    # TEST NETWORK 4
    test_number = 4
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_4 = network.Network((
        layers.FullyConnectedLayer(1, 2, activations.Sigmoid,
                                   testing=[1, 0]),

        layers.FullyConnectedLayer(2, 1, activations.ReLU,
                                   testing=[1, 0]),
    ))

    outputs = test_net_4.forward_pass([-2])
    expected = np.array([0.2384058], dtype=np.float32)
    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")

    # TEST NETWORK 5
    test_number = 5
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_5 = network.Network((
        layers.FullyConnectedLayer(1, 2, activations.Sigmoid,
                                   testing=[1, 1]),

        layers.FullyConnectedLayer(2, 1, activations.ReLU,
                                   testing=[1, 1]),
    ))

    outputs = test_net_5.forward_pass([1.234])
    expected = np.array([2.8065228], dtype=np.float32)
    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")

    # Time to test Convolution layers - This is going to cause pain

    # TEST NETWORK 6
    test_number = 6
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_6 = network.Network((
        layers.ConvolutedLayer((4, 4), (2, 2), 1, testing=[1, 0]),
    ))

    outputs = test_net_6.forward_pass([
        np.array([1, 1, 1, 1]),
        np.array([1, 1, 1, 1]),
        np.array([1, 1, 1, 1]),
        np.array([1, 1, 1, 1]),
    ])

    expected = np.array([4, 4, 4, 4], dtype=np.float32)
    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")

    # TEST NETWORK 7
    test_number = 7
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_7 = network.Network((
        layers.ConvolutedLayer((4, 4), (2, 2), 1, testing=[1, 0]),
    ))

    outputs = test_net_7.forward_pass([
        np.array([1, 2, 1, 2]),
        np.array([3, 4, 3, 4]),
        np.array([1, 2, 1, 2]),
        np.array([3, 4, 3, 4]),
    ])

    expected = np.array([10, 10, 10, 10], dtype=np.float32)
    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")

    # TEST NETWORK 8
    test_number = 8
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_8 = network.Network((
        layers.ConvolutedLayer((8, 8), (2, 2), 1, testing=[1, 0]),
        layers.ConvolutedLayer((4, 4), (2, 2), 1, testing=[1, 0])
    ))

    outputs = test_net_8.forward_pass([
        np.array([1, 2, 1, 2, 1, 2, 1, 2]),
        np.array([3, 4, 3, 4, 3, 4, 3, 4]),
        np.array([1, 2, 1, 2, 1, 2, 1, 2]),
        np.array([3, 4, 3, 4, 3, 4, 3, 4]),
        np.array([1, 2, 1, 2, 1, 2, 1, 2]),
        np.array([3, 4, 3, 4, 3, 4, 3, 4]),
        np.array([1, 2, 1, 2, 1, 2, 1, 2]),
        np.array([3, 4, 3, 4, 3, 4, 3, 4]),
    ])

    expected = np.array([40, 40, 40, 40], dtype=np.float32)

    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")

    # TEST NETWORK 9
    test_number = 9
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_9 = network.Network((
        layers.ConvolutedLayer((8, 8), (2, 2), 1, testing=[1, 0]),
        layers.FullyConnectedLayer(16, 2, activations.ReLU, testing=[0.5, 1])
    ))

    outputs = test_net_9.forward_pass([
        np.array([2, 1, 2, 1, 2, 1, 2, 1]),
        np.array([4, 3, 4, 3, 4, 3, 4, 3]),
        np.array([2, 1, 2, 1, 2, 1, 2, 1]),
        np.array([4, 3, 4, 3, 4, 3, 4, 3]),
        np.array([2, 1, 2, 1, 2, 1, 2, 1]),
        np.array([4, 3, 4, 3, 4, 3, 4, 3]),
        np.array([2, 1, 2, 1, 2, 1, 2, 1]),
        np.array([4, 3, 4, 3, 4, 3, 4, 3]),
    ])

    expected = np.array([81, 81], dtype=np.float32)

    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")

    # TEST NETWORK 10
    test_number = 10
    print(f"Test Net {test_number} | TESTING", end="")
    test_net_10 = network.Network((
        layers.ConvolutedLayer((8, 8), (2, 2), 1, testing=[1, 0]),
        layers.FullyConnectedLayer(16, 36, activations.ReLU, testing=[0.5, 1]),
        layers.ConvolutedLayer((6, 6), (2, 2), 1, testing=[1, 2])
    ))

    outputs = test_net_10.forward_pass([
        np.array([2, 1, 2, 1, 2, 1, 2, 1]),
        np.array([4, 3, 4, 3, 4, 3, 4, 3]),
        np.array([2, 1, 2, 1, 2, 1, 2, 1]),
        np.array([4, 3, 4, 3, 4, 3, 4, 3]),
        np.array([2, 1, 2, 1, 2, 1, 2, 1]),
        np.array([4, 3, 4, 3, 4, 3, 4, 3]),
        np.array([2, 1, 2, 1, 2, 1, 2, 1]),
        np.array([4, 3, 4, 3, 4, 3, 4, 3]),
    ])

    expected = np.array([326, 326, 326, 326, 326, 326, 326, 326, 326], dtype=np.float32)

    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net {test_number} | PASSED")
    else:
        print(f"\rTest Net {test_number} | FAILED | Output: {outputs}, Expected: {expected}")

    print("\n")

def test_output_validity_advanced():
    print("> VALIDATION TEST ADVANCED <")

    # TEST NETWORK 1
    test_number = 1
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_1 = network.Network((
        layers.FullyConnectedLayer(2, 3, activations.ReLU, loading=True),
        layers.FullyConnectedLayer(3, 2, activations.ReLU, loading=True)
    ))

    test_net_1.layout[0].weights = buffers.NetworkBuffer(np.array([0, 0, 2, 2, 1, 4], dtype=np.float32), (6,))
    test_net_1.layout[0].biases = buffers.NetworkBuffer(np.array([0, 2, -5], dtype=np.float32), (3,))

    test_net_1.layout[1].weights = buffers.NetworkBuffer(np.array([0, 0, 2, 0, 3, 0], dtype=np.float32), (6,))
    test_net_1.layout[1].biases = buffers.NetworkBuffer(np.array([-1, 1], dtype=np.float32), (2,))

    outputs = test_net_1.forward_pass([1, 2])  # [0, 8, 4]
    expected = np.array([7, 25], dtype=np.float32)
    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")


    # TEST NETWORK 2
    test_number = 2
    print(f"Test Net  {test_number} | TESTING", end="")
    test_net_2 = network.Network((
        layers.ConvolutedLayer((4, 4), (2, 2), 1, loading=True),
    ))

    test_net_2.layout[0].weights = [buffers.NetworkBuffer(np.array([12, 6, 4, 3], dtype=np.float32), (4,))]
    test_net_2.layout[0].biases = [buffers.NetworkBuffer(np.array([2], dtype=np.float32), (1,))]

    outputs = test_net_2.forward_pass([
        np.array([1, 2, 1, 2]),
        np.array([3, 4, 3, 4]),
        np.array([1, 2, 1, 2]),
        np.array([3, 4, 3, 4]),
    ])

    expected = np.array([50, 50, 50, 50], dtype=np.float32)
    if np.allclose(expected, outputs, atol=1e-6):
        print(f"\rTest Net  {test_number} | PASSED")
    else:
        print(f"\rTest Net  {test_number} | FAILED | Output: {outputs}, Expected: {expected}")

    print("\n")


if __name__ == "__main__":
    test_net = network.Network((
        layers.ConvolutedLayer((100, 100), (5, 5), filter_count=3, colour_depth=3),
        layers.FullyConnectedLayer(20*20*3, 2, activations.ReLU)
    ))

    #test_net = network.Load("net.pyn")

    test_performance(test_net)
    test_output_validity_basic()
    test_output_validity_advanced()
    test_accuracy(test_net)
