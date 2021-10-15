import unittest
from luxai21.models.base_nets.InceptionNet import InceptionNet_v1, InceptionNet_v2
from torchsummary import summary


import sys
sys.path.append("..")


class MyTestCase(unittest.TestCase):
    def test_InceptionNet_v1(self):
        net = InceptionNet_v1(18)
        summary(net, (18, 32, 32))

    def test_InceptionNet_v2(self):
        net = InceptionNet_v2(18)
        summary(net, (18, 32, 32))


if __name__ == '__main__':
    unittest.main()
