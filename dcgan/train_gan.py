#*-coding: utf-8 -*
import argparse
import os
import logging
from logging import getLogger, StreamHandler, Formatter
import chainer
from chainer import training
from chainer.training import extensions
from gan import Discriminator, Generator
from updater import GANUpdater

def set_logger():
  logger = getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  stream_handler = StreamHandler()
  stream_handler.setLevel(logging.DEBUG)
  # handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  # stream_handler.setFormatter(handler_format)
  logger.addHandler(stream_handler)
  logger.debug('Complete setting logger')
  return logger

def main():
  parser = argparse.ArgumentParser(description='DCGAN for mnist')
  parser.add_argument('--batchsize', '-b', type=int, default=1,
                      help='# of each mini-batch size')
  parser.add_argument('--epoch', '-e', type=int, default=500,
                      help='# of epoch')
  parser.add_argument('--gpu', '-g', type=int, default=-1,
                      help='GPU ID (if you want to use gpu, set positive value)')
  parser.add_argument('--dataset', '-d', type=str, default='',
                      help='path of training dataset path.')
  parser.add_argument('--out_dir', '-o', type=str, default='result',
                      help='path of output the result.')
  parser.add_argument('--n_hidden', '-n', type=int, default=100,
                      help='# of hidden unit(z)')
  args = parser.parse_args()
  
  logger = set_logger()

  logger.debug('='*10)
  logger.debug('GPU: {}'.format(args.gpu))
  logger.debug('#batchsize: {}'.format(args.batchsize))
  logger.debug('#epoch: {}'.format(args.epoch))
  logger.debug('n_hidden: {}'.format(args.n_hidden))
  logger.debug('dataset: {}'.format(args.dataset))
  logger.debug('out_dir: {}'.format(args.out_dir))
  logger.debug('='*10)

  print()
  logger.debug('setup models')
  
  # Setup networks
  generator = Generator(z_dim=args.n_hidden)
  discriminator = Discriminator(z_dim=args.n_hidden)
  if args.gpu >= 0:
    chainer.backends.cuda.get_device_from_id(args.gpu).use()
    generator.to_gpu()
    discriminator.to_gpu()

  # Setup optimizers
  def make_optimizer(model, alpha=0.0002, beta1=0.5):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    optimizer.add_hook(
                chainer.optimizer_hooks.WeightDecay(0.0001), 'hook_dec')
    return optimizer
  opt_generator = make_optimizer(generator)
  opt_discriminator = make_optimizer(discriminator)

  if args.dataset =='':
    train, _ = chainer.datasets.get_mnist(withlabel=False, ndim=3, scale=255.)
  else:
    pass
  # Setup an iterator
  train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

  # Setup an Updater
  updater = GANUpdater(models=(generator, discriminator), iterator=train_iter,
                       optimizer={
                           'gen': opt_generator, 'dis': opt_discriminator},
                       device=args.gpu)
  # Setup a trainer
  trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out_dir)
  snapshot_interval = (1000, 'iteration')
  display_interval = (100, 'iteration')
  trainer.extend(
          extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
          trigger=snapshot_interval)
  trainer.extend(extensions.snapshot_object(
          generator, 'generator_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
  trainer.extend(extensions.snapshot_object(
          discriminator, 'discriminator_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
  trainer.extend(extensions.LogReport(trigger=snapshot_interval))
  trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'gen/loss', 'dis/loss',
            ]), trigger=display_interval)
  trainer.extend(extensions.ProgressBar(update_interval=10))
  
  # Start train
  logger.debug('Training Start.')
  print()
  print()
  trainer.run()
  
if __name__ == '__main__':
  main()

