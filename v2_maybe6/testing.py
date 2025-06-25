from myconet.layer.fully_connected import FullyConnected
from myconet.layer.convoluted import Convoluted
from myconet.network import Network

from sampler.data_maker import load_training_data, get_cache_dir


# fixme - Backprop / Full Pop -> Not transferring back errors correctly?

# todo  - Add Backprop / input reduction


net = Network((
    Convoluted((100, 100, 3), (5, 5), 2, 1),  # ReLU
    FullyConnected(144, 1, 2),  # Sigmoid
), log_level=2)


print("(Might Be) Loading Cache From:", get_cache_dir())
data = load_training_data(
    "trainingPoints.txt",
    mutations_per_image=100,
    transformations_per_image=4,
    pre_load=False,
    cache=False,
)

print("loaded data")
test_sample = data[3]

print("Score (Pre):", net.score(test_sample, test_sample.output))

net.train(data[:2], [], 1, 0.1)

print("Score (Aft):", net.score(test_sample, test_sample.output))

net.release()