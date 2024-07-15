import os
import sys
import unittest
import numpy as np
from itertools import product
from nn.layers import Linear

sys.path.append("..")


class Test(unittest.TestCase):
    def test_Linear(self):
        for in_dim, out_dim, n_batch  in product((1, 10), (1, 10), range(1, 10)):
            linear = Linear(in_dim, out_dim)
            x = np.random.rand(n_batch, in_dim)
            y = linear.forward(x)
            self.assertEqual(y.shape, (in_dim, out_dim))