import matplotlib.pyplot as plt

import layers


class viewer:
    def __init__(self):
        self.fig = None
        self.axes = None

    def display(self, network, inputs):
        max_y = max(
            [layer.get_filter_count() if isinstance(layer, layers.ConvolutedLayer) else 1 for layer in network.layout])

        if self.fig is None:
            self.fig, self.axes = plt.subplots(max_y, len(network.layout), figsize=(10, 10))

        for ax in self.axes:
            for row in ax:
                row.clear()

        data = network.forward_pass(inputs, for_display=True)

        for i, dat in enumerate(data):
            if isinstance(network.layout[i], layers.FullyConnectedLayer):
                self.axes[max_y // 2][i].imshow([dat.get_as_array()], cmap='plasma', interpolation='nearest')

            elif isinstance(network.layout[i], layers.ConvolutedLayer):
                for ci, chunk in enumerate(dat):
                    chunk_data = chunk.get_as_array().reshape(network.layout[i].get_output_shape())

                    self.axes[ci][i].imshow(chunk_data, cmap='plasma', interpolation='nearest', vmax=1.5, vmin=-1.5)

        plt.tight_layout()
        plt.draw()  # Use plt.draw() to update the figure
        plt.pause(0.1)  # Pause briefly to allow updates
