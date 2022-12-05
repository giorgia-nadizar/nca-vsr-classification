import random
import re
import sys

import matplotlib.pylab as pl
import numpy as np
import tensorflow as tf
import tqdm

from nca import Model
from utils import PicklePersist


def parse_shape(line: str, width: int = 9, height: int = 4):
  string = re.sub(r'^.*?\[', '[', re.sub(r'^.*?=', '', line)).replace(' ', '').replace('[[', '') \
    .replace(']]', '').replace('\n', '').replace(',', '').replace('][', '-')
  tokens = string.split('-')
  if len(tokens) < 1:
    return None
  shape = []
  for token in tokens:
    row = []
    for number in token:
      if number == '0':
        row.append(0)
      else:
        row.append(1)
    shape.append(row)
  for row in shape:
    while len(row) < width:
      row.append(0)
  while len(shape) < height:
    shape.insert(0, [0 for _ in range(width)])
  return shape


def load_shapes_from_file(filename: str):
  shapes = []
  lines = open(filename, 'r').readlines()
  counter = 0
  for line in lines:
    shape = parse_shape(line)
    counter += 1
    if shape:
      shapes.append(shape)
  return shapes


def expand_y_label(x, y):
  y_res = np.zeros(list(x.shape) + [len(x)])
  # broadcast y to match x shape:
  y_expanded = np.broadcast_to(y, x.T.shape).T
  y_res[x >= 0.1, y_expanded[x >= 0.1]] = 1.0
  return y_res.astype(np.float32)


def train(x_train: list[list[list]], num_iterations: int = 1500, plots: bool = False):
  model = Model.standard_model(len(x_train))
  x_train = np.array(x_train).astype(np.float32)
  y_train = np.array(list(range(len(x_train))))
  y_train = expand_y_label(x_train, y_train)

  trainer = tf.keras.optimizers.Adam()
  losses = []
  accs = []

  x0 = tf.constant(x_train)  # (bs, y, x)
  mask = tf.expand_dims(x0, axis=3)  # (bs, y, x, 1)
  y0 = tf.constant(y_train)  # (bs, y, x, num_classes)

  for _ in tqdm.tqdm(range(num_iterations)):
    state = model.initialize(x0)
    with tf.GradientTape() as tape:
      for j in range(random.randint(9, 29)):
        update = model(state)  # (bs, 5, 4, n)
        random_mask = tf.cast(tf.random.uniform(state.shape) < 0.5, tf.float32)
        update = update * mask * random_mask  # mask out updates to "dead" cells (bs, y, x, n)
        state = state + update  # (bs, y, x, n)
      x = model.classify(state)  # (1, y, x, n_classes)
      loss = tf.reduce_mean((y0 - x) ** 2)

    grads = tape.gradient(loss, model.trainable_weights)
    trainer.apply_gradients(zip(grads, model.trainable_weights))
    model.reset_diag_kernel()

    losses.append(loss)

    y_label = tf.argmax(y0, axis=-1)  # (bs, y, x)
    x_label = tf.argmax(x, axis=-1)  # (bs, y, x)
    correct = tf.cast(tf.equal(x_label, y_label), tf.float32)  # (bs, y, x)
    acc = tf.reduce_sum(correct * x0) / tf.reduce_sum(x0)
    accs.append(acc.numpy().item())

  if plots:
    pl.figure(figsize=(10, 4))
    pl.title('loss')
    pl.xlabel('Number of steps')
    pl.ylabel('loss')
    pl.plot(losses, label="ca")
    pl.legend()
    pl.show()

    pl.figure(figsize=(10, 4))
    pl.title('accs')
    pl.xlabel('Number of steps')
    pl.ylabel('accs')
    pl.plot(accs, label="ca")
    pl.legend()
    pl.show()

  return model, losses, accs


def train_and_pickle(set_number: int, num_iterations: int = 1500, save_progress=True):
  shapes = load_shapes_from_file('shapes/sample_creatures_set' + str(set_number) + '.txt')
  model, losses, accs = train(shapes, num_iterations, plots=False)

  if save_progress:
    with open(f'training/progress_{set_number}.txt', 'w') as f:
      f.write('iteration;loss;acc\n')
      for iteration in range(num_iterations):
        f.write(f'{iteration};{losses[iteration].numpy()};{accs[iteration]}\n')

  perceive_kernel, pb = model.perceive.layers[0].get_weights()
  dk1, db1 = model.dmodel.layers[0].get_weights()
  dk2, db2 = model.dmodel.layers[1].get_weights()
  dictionary = {
    'dmodel_bias_1': db1,
    'dmodel_bias_2': db2,
    'dmodel_kernel_1': dk1[0][0][:][:],
    'dmodel_kernel_2': dk2[0][0][:][:],
    'perceive_bias': pb,
    'pk_bottom': perceive_kernel[2][1][:][:],
    'pk_left': perceive_kernel[1][0][:][:],
    'pk_right': perceive_kernel[1][2][:][:],
    'pk_self': perceive_kernel[1][1][:][:],
    'pk_top': perceive_kernel[0][1][:][:]
  }
  PicklePersist.compress_pickle('parameters/params_set' + str(set_number), data=dictionary)


if __name__ == '__main__':
  target_set = 1
  n_iterations = 1500

  args = sys.argv[1:]
  for arg in args:
    if arg.startswith('set'):
      target_set = int(arg.replace('set=', ''))
    if arg.startswith('n_it'):
      n_iterations = int(arg.replace('n_it=', ''))

  train_and_pickle(target_set, n_iterations)
