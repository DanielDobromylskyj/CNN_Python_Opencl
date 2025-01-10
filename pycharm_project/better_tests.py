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


class bcolors:
    HEADER = '\033[95m'
    PASS = '\033[92m'
    FAIL = '\033[91m'
    NO_TEST = '\033[96m'
    ENDC = '\033[0m'


def create_layer_from_small_layout(layer_type, args):
    if layer_type == "full":
        return layers.FullyConnectedLayer(*args)
    elif layer_type == "conv":
        return layers.ConvolutedLayer(*args)

    else:
        raise NotImplementedError(f"Failed to create network from small layout, invalid layer type '{layer_type}'")


def test_forward_output_validation(layout_small, inputs, expected):
    test_net = network.Network(tuple([
        create_layer_from_small_layout(layer_type, args)
        for layer_type, args in layout_small
    ]))

    outputs = test_net.forward_pass(inputs)
    expected = np.array(expected, dtype=np.float32)

    if np.allclose(expected, outputs, atol=1e-6):
        return [True, None]
    else:
        return [False, f"Output: {outputs}, Expected: {expected}"]


def test_forward_output_validation_complex(layout_small, data, inputs, expected):
    test_net = network.Network(tuple([
        create_layer_from_small_layout(layer_type, args)
        for layer_type, args in layout_small
    ]))

    for i, layer in enumerate(test_net.layout):
        layer.weights = data[i][0]
        layer.biases = data[i][1]

    outputs = test_net.forward_pass(inputs)
    expected = np.array(expected, dtype=np.float32)

    if np.allclose(expected, outputs, atol=1e-6):
        return [True, None]
    else:
        return [False, f"Output: {outputs}, Expected: {expected}"]


def test_backprop_efficiency(layout_small, data, training_data, epoches, learning_rate, max_total_error):
    test_net = network.Network(tuple([
        create_layer_from_small_layout(layer_type, args)
        for layer_type, args in layout_small
    ]))

    for i, layer in enumerate(test_net.layout):
        layer.weights = data[i][0]
        layer.biases = data[i][1]

    test_net.train(training_data, training_data, epoches, learning_rate, show_stats=False)

    error = 0
    for sample, target in training_data:
        outputs = test_net.forward_pass(sample)
        error += sum([abs(outputs[i] - target[i]) for i in range(len(target))]) / len(outputs)

    error /= len(training_data)

    if error < max_total_error:
        return [True, None]
    else:
        return [False, f"Max Error Too High. Wanted: {max_total_error}, Got {error}"]


def test_backward_gradient_calcs(layout_small, network_data, inputs, target, learning_rate, expected_grads):
    test_net = network.Network(tuple([
        create_layer_from_small_layout(layer_type, args)
        for layer_type, args in layout_small
    ]))

    for i, layer in enumerate(test_net.layout):
        layer.weights = network_data[i][0]
        layer.biases = network_data[i][1]

    gradients = test_net.backward_pass(inputs, target, learning_rate)

    for layer_index, [weights, biases] in enumerate(expected_grads):
        gradients_weights = gradients[layer_index][0].get_as_array()
        gradients_biases = gradients[layer_index][1].get_as_array()

        errors = []

        for i, weight in enumerate(weights):
            if round(weight, 6) != round(gradients_weights[i], 6):
                errors.append(f"Layer {layer_index}, Weight {i}, Expected {weight}, Got {gradients_weights[i]}")

        for i, bias in enumerate(biases):
            if round(bias, 6) != round(gradients_biases[i], 6):
                errors.append(f"Layer {layer_index}, Bias {i}, Expected {bias}, Got {gradients_biases[i]}")

    if not errors:
        return [True, None]

    if len(errors) > 1:
        return [False, f"{errors[0]} (+{len(errors)-1} more)"]

    return [False, errors[0]]


def perform_tests():
    tests = {
        "Output Validation (Full)": [
            (test_forward_output_validation, (
                [("full", (5, 3, activations.Sigmoid, False, [1, 0])),
                 ("full", (3, 2, activations.Sigmoid, False, [1, 0]))],
                [1, 2, 3, 4, 5],
                [0.952574, 0.952574]
            )),

            (test_forward_output_validation, (
                [("full", (1, 2, activations.Sigmoid, False, [1, 0])),
                 ("full", (2, 1, activations.ReLU, False, [1, 0]))],
                [1],
                [1.4621172]
            )),

            (test_forward_output_validation, (
                [("full", (1, 2, activations.Sigmoid, False, [1, 0])),
                 ("full", (2, 1, activations.ReLU, False, [1, 0]))],
                [5.3],
                [1.9900664]
            )),

            (test_forward_output_validation, (
                [("full", (1, 2, activations.Sigmoid, False, [1, 0])),
                 ("full", (2, 1, activations.ReLU, False, [1, 0]))],
                [-2],
                [0.2384058]
            )),

            (test_forward_output_validation, (
                [("full", (1, 2, activations.Sigmoid, False, [1, 1])),
                 ("full", (2, 1, activations.ReLU, False, [1, 1]))],
                [1.234],
                [2.8065228]
            )),

            (test_forward_output_validation, (
                [("full", (1, 2, activations.Sigmoid, False, [1, 1])),
                 ("full", (2, 1, activations.ReLU, False, [1, 1]))],
                [1.234],
                [2.8065228]
            )),
        ],

        "Output Validation (Conv)": [
            (test_forward_output_validation, (
                [("conv", ((4, 4), (2, 2), 1, 1, False, [1, 0]))],
                [
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                    np.array([1, 1, 1, 1]),
                ],
                [4, 4, 4, 4]
            )),

            (test_forward_output_validation, (
                [("conv", ((4, 4), (2, 2), 1, 1, False, [1, 0]))],
                [
                    np.array([1, 2, 1, 2]),
                    np.array([3, 4, 3, 4]),
                    np.array([1, 2, 1, 2]),
                    np.array([3, 4, 3, 4]),
                ],
                [10, 10, 10, 10]
            )),

            (test_forward_output_validation, (
                [
                    ("conv", ((8, 8), (2, 2), 1, 1, False, [1, 0])),
                    ("conv", ((4, 4), (2, 2), 1, 1, False, [1, 0])),
                ],
                [
                    np.array([1, 2, 1, 2, 1, 2, 1, 2]),
                    np.array([3, 4, 3, 4, 3, 4, 3, 4]),
                    np.array([1, 2, 1, 2, 1, 2, 1, 2]),
                    np.array([3, 4, 3, 4, 3, 4, 3, 4]),
                    np.array([1, 2, 1, 2, 1, 2, 1, 2]),
                    np.array([3, 4, 3, 4, 3, 4, 3, 4]),
                    np.array([1, 2, 1, 2, 1, 2, 1, 2]),
                    np.array([3, 4, 3, 4, 3, 4, 3, 4]),
                ],
                [40, 40, 40, 40]
            )),
        ],

        "Output Validation (Mix)": [
            (test_forward_output_validation, (
                [
                    ("conv", ((8, 8), (2, 2), 1, 1, False, [1, 0])),
                    ("full", (16, 2, activations.ReLU, False, [0.5, 1])),
                ],
                [
                    np.array([2, 1, 2, 1, 2, 1, 2, 1]),
                    np.array([4, 3, 4, 3, 4, 3, 4, 3]),
                    np.array([2, 1, 2, 1, 2, 1, 2, 1]),
                    np.array([4, 3, 4, 3, 4, 3, 4, 3]),
                    np.array([2, 1, 2, 1, 2, 1, 2, 1]),
                    np.array([4, 3, 4, 3, 4, 3, 4, 3]),
                    np.array([2, 1, 2, 1, 2, 1, 2, 1]),
                    np.array([4, 3, 4, 3, 4, 3, 4, 3]),
                ],
                [81, 81]
            )),

            (test_forward_output_validation, (
                [
                    ("conv", ((8, 8), (2, 2), 1, 1, False, [1, 0])),
                    ("full", (16, 36, activations.ReLU, False, [0.5, 1])),
                    ("conv", ((6, 6), (2, 2), 1, 1, False, [1, 2])),
                ],
                [
                    np.array([2, 1, 2, 1, 2, 1, 2, 1]),
                    np.array([4, 3, 4, 3, 4, 3, 4, 3]),
                    np.array([2, 1, 2, 1, 2, 1, 2, 1]),
                    np.array([4, 3, 4, 3, 4, 3, 4, 3]),
                    np.array([2, 1, 2, 1, 2, 1, 2, 1]),
                    np.array([4, 3, 4, 3, 4, 3, 4, 3]),
                    np.array([2, 1, 2, 1, 2, 1, 2, 1]),
                    np.array([4, 3, 4, 3, 4, 3, 4, 3]),
                ],
                [326, 326, 326, 326, 326, 326, 326, 326, 326]
            )),
        ],

        "Advanced Validation (Mix)": [
            (test_forward_output_validation_complex, (
                [
                    ("full", (2, 3, activations.ReLU)),
                    ("full", (3, 2, activations.ReLU)),
                ],

                [
                    [
                        buffers.NetworkBuffer(np.array([0, 0, 2, 2, 1, 4], dtype=np.float32), (6,)),
                        buffers.NetworkBuffer(np.array([0, 2, -5], dtype=np.float32), (3,))
                    ],
                    [
                        buffers.NetworkBuffer(np.array([0, 0, 2, 0, 3, 0], dtype=np.float32), (6,)),
                        buffers.NetworkBuffer(np.array([-1, 1], dtype=np.float32), (2,))
                    ],
                ],

                [1, 2],
                [7, 25]
            )),

            (test_forward_output_validation_complex, (
                [
                    ("conv", ((4, 4), (2, 2), 1)),
                ],
                [
                    [
                        [buffers.NetworkBuffer(np.array([12, 6, 4, 3], dtype=np.float32), (4,))],
                        [buffers.NetworkBuffer(np.array([2], dtype=np.float32), (1,))]
                    ],
                    [
                        buffers.NetworkBuffer(np.array([0, 0, 2, 0, 3, 0], dtype=np.float32), (6,)),
                        buffers.NetworkBuffer(np.array([-1, 1], dtype=np.float32), (2,))
                    ],
                ],
                [
                    np.array([1, 2, 1, 2]),
                    np.array([3, 4, 3, 4]),
                    np.array([1, 2, 1, 2]),
                    np.array([3, 4, 3, 4]),
                ],
                [50, 50, 50, 50]
            )),
        ],
        "Backprop Validation (Full)": [
            (test_backward_gradient_calcs, (
                [
                    ("full", (2, 3, activations.ReLU)),
                    ("full", (3, 2, activations.ReLU)),
                ],

                [
                    [
                        buffers.NetworkBuffer(np.array([0, 0, 2, 2, 1, 4], dtype=np.float32), (6,)),
                        buffers.NetworkBuffer(np.array([0, 2, -5], dtype=np.float32), (3,))
                    ],
                    [
                        buffers.NetworkBuffer(np.array([0, 0, 2, 0, 3, 0], dtype=np.float32), (6,)),
                        buffers.NetworkBuffer(np.array([-1, 1], dtype=np.float32), (2,))
                    ],
                ],

                [1, 0.5],  # inputs
                [0.5, 1],  # targets
                0.1,  # learning rate

                [  # Expected gradients
                    [[], []],
                    [[0, 0, 0, -1, 0, 0], []],  # might be incorrect here (could be very. But IDK)
                ]
            )),

            (test_backward_gradient_calcs, (
                [
                    ("full", (2, 3, activations.ReLU)),
                    ("full", (3, 2, activations.ReLU)),
                ],

                [
                    [
                        buffers.NetworkBuffer(np.array([0, 0, 2, 2, 1, 4], dtype=np.float32), (6,)),
                        buffers.NetworkBuffer(np.array([0, 2, -5], dtype=np.float32), (3,))
                    ],
                    [
                        buffers.NetworkBuffer(np.array([0, 0, 2, 0, 3, 0], dtype=np.float32), (6,)),
                        buffers.NetworkBuffer(np.array([-1, 1], dtype=np.float32), (2,))
                    ],
                ],

                [1, 0.5],  # inputs
                [0.5, 1],  # targets
                0.01,  # learning rate

                [  # Expected gradients
                    [[], []],
                    [[0, 0, 0, -0.75, 0, 0], []],
                ]
            )),
        ],

        "Backprop Effectiveness (Full)": [
            (test_backprop_efficiency, (
                [
                    ("full", (1, 2, activations.ReLU)),
                ],

                [
                    [
                        buffers.NetworkBuffer(np.array([1, 1], dtype=np.float32), (2,)),
                        buffers.NetworkBuffer(np.array([0, 0], dtype=np.float32), (2,))
                    ]
                ],

                [[[1], [1, 3]], [[2], [3, 5]], [[3], [5, 7]]],
                40,
                0.01,  # learning rate
                .5
            )),

            (test_backprop_efficiency, (
                [
                    ("full", (1, 2, activations.ReLU)),
                ],

                [
                    [
                        buffers.NetworkBuffer(np.array([1, 1], dtype=np.float32), (2,)),
                        buffers.NetworkBuffer(np.array([0, 0], dtype=np.float32), (2,))
                    ]
                ],

                [[[1], [1, 1]], [[0.5], [1, 0.5]], [[0], [1, 0]], [[3], [1, 3]], [[5], [1, 5]]],
                65,
                0.02,  # learning rate
                .3
            )),

            (test_backprop_efficiency, (
                [
                    ("full", (2, 2, activations.ReLU, False, None, [0.9, 0.999])),
                ],

                [
                    [
                        buffers.NetworkBuffer(np.array([1, 1, 1, 1], dtype=np.float32), (4,)),
                        buffers.NetworkBuffer(np.array([0, 0], dtype=np.float32), (2,))
                    ]
                ],

                [[[1, 1], [2, 3]], [[2, 1], [4, 3]], [[3, 3], [6, 9]]],
                200,
                0.05,  # learning rate
                2.0
            )),
        ],
    }

    test_results = {}

    progress_bar_chunks = 40
    for j, sub_category in enumerate(tests):
        tests_in_cat = len(tests[sub_category])
        test_results[sub_category] = []

        for i, test in enumerate(tests[sub_category]):
            test_func, test_args = test

            result = test_func(*test_args)
            test_results[sub_category].append(result)

            print(
                f"\r{sub_category} | {round(((i + 1) / tests_in_cat) * 100)}% | {'#' * round((j / len(tests) * progress_bar_chunks))}{' ' * (progress_bar_chunks - round((j / len(tests) * progress_bar_chunks)))} |",
                end="")

    print(f"\r{sub_category} | {round(((i + 1) / tests_in_cat) * 100)}% | {'#' * progress_bar_chunks} |", end="")

    print("\n")

    max_results = max([len(tests[sub_category]) for sub_category in tests])
    max_label_width = max(max([len(sub_category) for sub_category in tests]), 8) + 1

    spacing = ' '
    print(f"Category{spacing * (max_label_width - 8)}| {' | '.join([f'Test {i + 1}{spacing * (len(str(max_results)) - len(str(i + 1)))}' for i in range(max_results)])} |")
    print(f"{'-' * max_label_width}+-{'-+-'.join(['-----' + ('-' * len(str(max_results))) for i in range(max_results)])}-+")
    for sub_category in tests:
        print(f"{sub_category}{spacing * (max_label_width - len(sub_category))}| {' | '.join([(f' {bcolors.PASS}PASS{bcolors.ENDC}' if result[0] is True else f' {bcolors.FAIL}FAIL{bcolors.ENDC}') + ' ' * len(str(max_results)) for result in test_results[sub_category]])} |", end="")
        if max_results - len(test_results[sub_category]) != 0:
            print(" " + " | ".join([f" {bcolors.NO_TEST}None{bcolors.ENDC} " for i in range(max_results - len(test_results[sub_category]))]) + " |")
        else:
            print("")

        print(f"{'-' * max_label_width}+-{'-+-'.join(['-----' + ('-' * len(str(max_results))) for i in range(max_results)])}-+")
        
    print("\n")  # 2 new lines

    all_failures = [(sub_category, i, result[1]) for i, result in enumerate(test_results[sub_category]) for sub_category in test_results if result[0] is False]

    if all_failures:
        print(f"{bcolors.FAIL}>> Failure Debug Output <<{bcolors.ENDC}\n")

        for sub_category in tests:
            failures = [(i+1, result) for i, result in enumerate(test_results[sub_category]) if result[0] is False]

            if failures:
                print(f"{bcolors.NO_TEST}> {sub_category}{bcolors.ENDC}")

                for test_index, failure in failures:
                    print(f"Test {test_index}: {failure[1]}")

                print("\n")


if __name__ == "__main__":
    perform_tests()
