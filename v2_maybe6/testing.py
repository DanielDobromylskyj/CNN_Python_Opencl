import numpy as np

from myconet.layer.fully_connected import FullyConnected
from myconet.network import Network

# I have royally fucked this project. Good luck future me. Im going to bed


net = Network((
    FullyConnected(1, 3, 0),
    FullyConnected(3, 2, 0),
))


output = net.forward(np.array([2]))


