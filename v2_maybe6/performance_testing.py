from myconet.network import Network
from myconet.layer.fully_connected import FullyConnected
from myconet.layer.convoluted import Convoluted

from tabulate import tabulate
import numpy as np
import random
import time

test_data = [
    np.random.randn(30_000).astype(np.float32)
    for _ in range(1000)
]

print("Creating Training Data...  (50%)")

training_data = [
    (
        data,
        np.array((random.randint(0, 1),), dtype=np.float32)
     )
    for data in test_data
]

print("Created training Data")

net = Network((
    Convoluted((100, 100, 3), (5, 5), 2, 1),  # ReLU
    FullyConnected(2304, 1, 2),  # Sigmoid
), log_level=1)


# Forward Test -> Standard Execution

standard_start = time.time()


outputs1 = [
    net.forward(item, batch=False) for item in test_data
]

standard_end = time.time()

print("Completed Forward Standard Tests")

# Forward Test -> Batch Execution

batched_start = time.time()

outputs2 = net.forward(test_data, batch=True)

batched_end = time.time()

print("Completed Forward Batched Tests")

# Training Test -> Standard Execution
net.force_load_all_kernels()  # Force load kernels as we are not going though normal .train() method

training_standard_start = time.time()

"""for inputs, outputs in training_data:
    net.backward(inputs, outputs, 0.01)"""

training_standard_end = time.time()

# Training Test -> Batch Execution

training_batch_start = time.time()

# todo

training_batch_end = time.time()

# Test Output / Calculations

forward_standard_elapsed = standard_end - standard_start
forward_batched_elapsed = batched_end - batched_start

backward_standard_elapsed = training_standard_end - training_standard_start
backward_batch_elapsed = training_batch_end - training_batch_start


print(outputs1[:5])
print(outputs2[:5])

output_match = all([
    round(outputs1[i][0], 4) == round(outputs2[i][0], 4)
    for i in range(len(outputs1))
]) if len(outputs1) == len(outputs2) else "Different Sizes!"

print(">> PERFORMANCE BREAK DOWN <<")
print("Batch Size ->", len(test_data))
print(f"Batch Speed Increase -> {round((forward_standard_elapsed - forward_batched_elapsed) / forward_standard_elapsed * 100, 1)}%")
print("Output Match ->", output_match)
print("\n")
data = [
    ["Standard", "Forward", "Iterative", round(forward_standard_elapsed, 3), round(forward_standard_elapsed / len(test_data) * 1000, 2)],
    ["Batched", "Forward", "Grouped", round(forward_batched_elapsed, 3), round(forward_batched_elapsed / len(test_data) * 1000, 2)],

    ["Standard", "Backward", "Iterative", round(backward_standard_elapsed, 3),
     round(backward_standard_elapsed / len(test_data) * 1000, 2)],
    ["Batched", "Backward", "Grouped", round(backward_batch_elapsed, 3), round(backward_batch_elapsed / len(test_data) * 1000, 2)],
]

headers = ["Name", "Direction", "Type", "Total Time (s)", "Time Per Item (ms)"]

net.log.disable()
print(tabulate(data, headers=headers, tablefmt="grid"))
net.log.enable()


