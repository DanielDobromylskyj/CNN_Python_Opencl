'''from myconet.layer.fully_connected import FullyConnected
from myconet.network import Network

net = Network((
    FullyConnected(100, 50, 0),
    FullyConnected(50, 2, 0),
))


net.save("testnet.pyn")

net2 = Network.load("testnet.pyn")
print(net2.layout)

'''

from sampler.data_maker import load_training_data, get_cache_dir

print(get_cache_dir())
data = load_training_data(
    "trainingPoints.txt",
    mutations_per_image=100,
    transformations_per_image=4,
    pre_load=True,
    cache=True,
)

