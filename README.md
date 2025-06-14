# 🦠 MycoNet


> **Disclaimer:** This project is still in development and may break or behave unexpectedly. Use at your own risk!



## Overview

MycoNet is a neural networking tool with intergrated GPU acceleration using OpenCL 2.0, Allowing it to run quickly on many devices, including AMD and NVIDEA cards!
MycoNet is a tool produced to train and process neural networks for my Find-A-Bac project, Where I aim to help detect mycobacterium in animal tissue using AI.

> **Disclaimer:** As of 14/06/25 | Only tested on a AMD 7600 XT & Radeon 610M (Not that the 610M was very happy)
>
> **Disclaimer 2:** Support is not guaranteed, contact me if you are having problems.

## Features

- Custom Network Support
- Convoluted Layer Suport
- Backpropagation Support
- Multiple Optimizers
- Custom Logging (With Levels)

## Examples

```python
from myconet.layer.fully_connected import FullyConnected
from myconet.layer.convoluted import Convoluted
from myconet.network import Network

#  As of 14/06/25, There is not yet a activations class / list
#  Only ReLU & Sigmoid are currently supported (With room to expand)

net = Network((
    Convoluted((100, 100, 3), (5, 5), 2, 1),  # ReLU
    FullyConnected(144, 1, 2),  # Sigmoid
), log_level=2)

net.save("my_neural_network.pyn")
net.release()  # Remove all buffers (And stop logger)

```

```python
from myconet.network import Network

net = Network.load("my_neural_network.pyn")

outputs = net.forward(input_data: np.ndarray)

net.release()

```

## 🛠️ Installation

Ensure requrements are met:
- Python 3.12
- OpenCL 2.0

Pip requirements:
- pyopencl
- openslide
- numpy

> **Warning:** This list may not be exhaustive, as project is still in development

## About Me

This project is maintained by [Daniel Dobromylskyj](https://github.com/DanielDobromylskyj). You can reach me at daniel.dobromylskyj@outlook.com.

## 📝 License

[Apache](LICENSE)
