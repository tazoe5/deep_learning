# *- coding:utf-8 -*
import chainer
import chainer.functions as F
from chainer.training.updater import StandardUpdater

class GANUpdater(StandardUpdater):
  def __init__(self, *args, **kwargs):
    self.generator, self.discriminator = kwargs.pop('models')
    super().__init__(*args, **kwargs)
  
  def loss_discriminator(self, discriminator, x_fake, x_real):
    batchsize = len(x_fake)
    loss1 = F.sum(F.softplus(-x_real)) / batchsize
    loss2 = F.sum(F.softplus(x_fake)) / batchsize
    loss = loss1 + loss2
    chainer.report({'loss': loss}, discriminator)
    return loss
  
  def loss_generator(self, generator, x_fake):
    batchsize = len(x_fake)
    loss = F.sum(F.softplus(-x_fake)) / batchsize
    chainer.report({'loss': loss}, generator)
    return loss
  
  def update_core(self):
    gen_optimizer = self.get_optimizer('gen')
    dis_optimizer = self.get_optimizer('dis')

    batch = self.get_iterator('main').next()
    x_real = chainer.Variable(self.converter(batch, self.device)) / 255.
    xp = chainer.backends.cuda.get_array_module(x_real.array)

    generator, discriminator = self.generator, self.discriminator
    batchsize = len(batch)

    # detect image
    dis_real = discriminator(x_real)
    z = chainer.Variable(xp.asarray(generator.make_hidden(batchsize)))
    x_fake = generator(z)
    dis_fake = discriminator(x_fake)
    
    dis_optimizer.update(self.loss_discriminator, discriminator, dis_fake, dis_real)
    gen_optimizer.update(self.loss_generator, generator, dis_fake)


