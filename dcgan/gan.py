#*- coding:utf-8 -*
import os, sys
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
import numpy as np
from tqdm import tqdm

##### params #####
Ng = 128
Nz = 100
Mg = 16
Md = 4
Nd =128
W0, H0 = 64, 64
W, H =256, 256
##################
def add_noise(h, sigma=0.2):
  xp = chainer.backends.cuda.get_array_module(h.array)
  if chainer.config.train:
    return h + sigma * xp.random.randn(*h.shape)
  else:
    return h

"""
Implementation of DCGAN using chainer.

"""

class Generator(chainer.Chain):
  def __init__(self, z_dim, wscale=0.02):
    super().__init__()
    self.z_dim = z_dim
    with self.init_scope():
      w = chainer.initializers.Normal(wscale)
      self.l1 = L.Linear(z_dim, 256 * 3 * 3, initialW=w)
      self.dc1 = L.Deconvolution2D(256, 128, 2, stride=2, pad=1, initialW=w) 
      self.dc2 = L.Deconvolution2D(128, 64, 4, stride=4, pad=1, initialW=w)
      self.dc3 = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, initialW=w)

      self.bn1 = L.BatchNormalization(256)
      self.bn2 = L.BatchNormalization(128)
      self.bn3 = L.BatchNormalization(64)
  
  def make_hidden(self, batchsize):
    return cuda.cupy.random.uniform(-1, 1, (batchsize, self.z_dim, 1, 1))\
            .astype(np.float32)

  def __call__(self, z, test=False):
    h = self.l1(z)
    h = F.reshape(h, (z.data.shape[0], 256, 3, 3))
    h = F.relu(self.bn1(h))
    h = F.relu(self.bn2(self.dc1(h)))
    h = F.relu(self.bn3(self.dc2(h)))
    x = F.sigmoid(self.dc3(h))
    return x

class Discriminator(chainer.Chain):
  def __init__(self, z_dim, wscale=0.02):
    super().__init__()
    self.z_dim = z_dim
    with self.init_scope():
      w = chainer.initializers.Normal(wscale)
      self.c1 = L.Convolution2D(1, 64, 4, stride=2, pad=1, initialW=w)
      self.c2 = L.Convolution2D(64, 128, 4, stride=4, pad=1, initialW=w)
      self.c3 = L.Convolution2D(128, 256, 2, stride=2, pad=1, initialW=w)
      self.l1 = L.Linear(256 * 3 * 3, z_dim, initialW=w)

      self.bn1 = L.BatchNormalization(64)
      self.bn2 = L.BatchNormalization(128)
      self.bn3 = L.BatchNormalization(256)

  def __call__(self, x, test=False):
    h = add_noise(x)
    h = F.leaky_relu(add_noise(self.bn1(self.c1(h))))
    h = F.leaky_relu(add_noise(self.bn2(self.c2(h))))
    h = F.leaky_relu(add_noise(self.bn3(self.c3(h))))
    z = self.l1(h)
    return z

if __name__ == '__main__':
  
  z_dim = 100
  z_num = 3
  gpu = 0

  model_g = Generator(z_dim)
  model_d = Discriminator(z_dim)
  
  if gpu == 0:
    cuda.get_device(gpu).use()
    model_g.to_gpu(gpu)
    model_d.to_gpu(gpu)
    xp = cuda.cupy
  else:
    xp = np
  print('model setup.')
  z = xp.random.random((z_num, z_dim)).astype(xp.float32)
  z= chainer.Variable(z)
  for i in tqdm(np.arange(100000)):
    x = model_g(z)
    z_d = model_d(x)
  print('z.shape: {}'.format(z.shape))
  print('x.shape: {}'.format(x.shape))
  print('z_d.shape: {}'.format(z_d.shape))
