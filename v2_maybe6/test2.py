from myconet.layer.fully_connected import FullyConnected
from myconet.network import Network


net = Network((
    FullyConnected(1, 2, 1),
    FullyConnected(2, 1, 1),
), log_level=2)


net.save("test_pyn_spec.pyn")
net.release()


net2 = Network.load("test_pyn_spec.pyn")
print(net2)
