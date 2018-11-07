import sys, os
import chainer
from chainer import serializers
import numpy as np
from gan import Generator

if __name__ == '__main__':
    z_dim = 100
    model = Generator(z_dim)
    serializers.load_npz("result\generator_iter_999000.npz", model)
    print("load done")
    z = np.random.random(3, z_dim)
