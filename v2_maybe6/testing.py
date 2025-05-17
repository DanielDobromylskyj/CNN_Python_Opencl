from myconet.layer.fully_connected import FullyConnected
from myconet.network import Network

net = Network((
    FullyConnected(100, 50, 0),
    FullyConnected(50, 2, 0),
))


net.save("testnet.pyn")

net2 = Network.load("testnet.pyn")
print(net2.layout)

