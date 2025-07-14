from myconet.network import Network
from myconet.layer.fully_connected import FullyConnected
from myconet.layer.convoluted import Convoluted

import numpy as np
import time

test_data = [
    np.random.randn(30_000)
    for _ in range(2000)
]

print("Created training Data")

net = Network((
    Convoluted((100, 100, 3), (5, 5), 2, 1),  # ReLU
    FullyConnected(144, 1, 2),  # Sigmoid
), log_level=1)



standard_start = time.time()

for item in test_data:
    net.forward(item)

standard_end = time.time()

print("Completed Standard Tests")

batched_start = time.time()

net.forward(test_data, batch=True)

batched_end = time.time()

standard_elapsed = standard_end - standard_start
batched_elapsed = batched_end - batched_start

print(">> PERFORMANCE BREAK DOWN <<")
print("Batch Size:", len(test_data))
print()
print("Standard (Iterative): ")
print("Total Time (ms):", round(standard_elapsed * 1000, 1))
print("Per Item (ms):", round(standard_elapsed / len(test_data) * 1000, 1))
print()
print("Batched (Grouped): ")
print("Total Time (ms):", round(batched_elapsed * 1000, 1))
print("Per Item (ms):", round(batched_elapsed / len(test_data) * 1000, 1))
print()
print(f"Batched is {round((standard_elapsed - batched_elapsed) / standard_elapsed * 100, 1)}% faster than Standard")




