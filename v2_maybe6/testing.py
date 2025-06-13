from myconet.layer.fully_connected import FullyConnected
from myconet.layer.convoluted import Convoluted
from sampler.data_maker import load_training_data, get_cache_dir
from myconet.network import Network


# fixme - Backprop / Full Pop -> Not transferring back errors correctly?


net = Network((
    Convoluted((100, 100, 3), (5, 5), 2, 1),
    FullyConnected(144, 1, 1),
), log_level=3)


print(get_cache_dir())
data = load_training_data(
    "trainingPoints.txt",
    mutations_per_image=100,
    transformations_per_image=4,
    pre_load=False,
    cache=False,
)

print("loaded data")
test_sample = data[0]

print("Score (Pre):", net.score(test_sample, test_sample.output))

net.train(data[:2], [], 1, 0.1)

print("Score (Aft):", net.score(test_sample, test_sample.output))

