# Custom Convolutional Neural Network with OpenCL

## Overview

This is a custom Convolutional Neural Network (CNN) implementation built from scratch using OpenCL and Python (NumPy), created for my CS A-Level programming project. The project explores and demonstrates low-level neural network operations and how to run them efficiently on a GPU using OpenCL, rather than relying on libraries like TensorFlow or PyTorch.

## Features

- **Fully Custom CNN Implementation**: Custom CNN structure with forward pass and backpropagation.
- **GPU-accelerated computations**: Leverages OpenCL for GPU-based acceleration of matrix operations.
- **Flexible Configuration**: Supports custom architectures with different layers, kernel sizes, and activation functions.
- **Python 3.12 Compatibility**: Built and tested in Python 3.12 with NumPy for data handling.

## Why OpenCL?

OpenCL allows parallel computing across various hardware platforms, including GPUs. By using OpenCL, this project achieves hardware-accelerated performance, making it suitable for CNN tasks that are computationally intensive.

## Requirements

- **Python 3.12**
- **OpenCL-compatible GPU and drivers**
- **Dependencies**:
  - `pyopencl`
  - `numpy`
  - `matplotlib` (for visualizations, optional)

Install dependencies with:
```bash
pip install pyopencl numpy matplotlib
```

## Usage

1. **Define the CNN Architecture**: Configure the layers and settings for your CNN in the main script.

2. **Making A Model**: Use the `Network()` class to make a new model. For example:

   ```python
   from CNN.network import Network
   from CNN.activations import ReLU
   from CNN.layers import *

   model = Network(
       layout=(
           ConvolutedLayer(input_size=(100, 100), kernel_size=(5,5), filter_count=8, colour_depth=1),  # colour depth is just the number of colour channels your input has
           FullyConnectedLayer(input_size=3200, output_size=2, activation=ReLU)
       )
   )
   ```
