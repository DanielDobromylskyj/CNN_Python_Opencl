from myconet.layer.fully_connected import FullyConnected
from sampler.data_maker import load_training_data, get_cache_dir
from myconet.network import Network

net = Network((
    FullyConnected(2, 2, 1),
    FullyConnected(2, 1, 1),
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
out = net.forward([4, 1])
print("Output:", out)


