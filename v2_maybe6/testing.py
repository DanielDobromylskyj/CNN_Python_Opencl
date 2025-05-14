from myconet.layer import fully_connected
from myconet.network import Network

net = Network((
    fully_connected.FullyConnected(100, 100, 0),
))


net.save("testnet.pyn")

net2 = Network.load("testnet.pyn")
print(net2.layout)

