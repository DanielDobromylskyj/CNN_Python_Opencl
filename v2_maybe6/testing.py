from myconet.layer.fully_connected import FullyConnected
from sampler.data_maker import load_training_data, get_cache_dir
from myconet.network import Network

net = Network((
    FullyConnected(1, 1, 1),
    FullyConnected(1, 1, 1),
))


print(get_cache_dir())
data = load_training_data(
    "trainingPoints.txt",
    mutations_per_image=100,
    transformations_per_image=4,
    pre_load=False,
    cache=False,
)

print("loaded data")

net.train(3,3)  # bodge, to load Training kernel

out = net.backward([1], [2], 0.1)
print("Output:", out)


