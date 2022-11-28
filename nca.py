from __future__ import annotations

import csv
import time
from multiprocessing import RawArray

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D

from utils import PicklePersist


class Model(tf.keras.Model):

  @classmethod
  def standard_model(cls, class_n: int):
    channel_n = 21
    grid_size_x = 9
    grid_size_y = 4
    return Model(channel_n, class_n, grid_size_x, grid_size_y)

  def __init__(self, channel_n: int, class_n: int, grid_size_x: int, grid_size_y: int):
    super().__init__()
    self.channel_n = channel_n
    self.class_n = class_n
    self.grid_size_x = grid_size_x
    self.grid_size_y = grid_size_y
    self.kernel_mask = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])[:, :, np.newaxis, np.newaxis]
    self.perceive = tf.keras.Sequential([
      Conv2D(30, 3, activation=tf.nn.relu, padding="SAME"),  # (c, 3, 3, 80)
    ])
    self.dmodel = tf.keras.Sequential([
      Conv2D(30, 1, activation=tf.nn.relu),  # (80, 1, 1, 80)
      Conv2D(self.channel_n, 1, activation=None, kernel_initializer=tf.zeros_initializer),  # (80, 1, 1, c)
    ])
    self(tf.zeros([self.class_n, self.grid_size_y, self.grid_size_x, self.channel_n]))  # dummy calls to build the model
    self.reset_diag_kernel()

  def reset_diag_kernel(self):
    kernel, bias = self.perceive.layers[0].get_weights()
    self.perceive.layers[0].set_weights([kernel * self.kernel_mask, bias])

  @tf.function
  def call(self, x):
    ds = self.dmodel(self.perceive(x))
    return ds

  @tf.function
  def classify(self, x):
    # The last x layers are the classification predictions, one channel per class.
    # Keep in mind there is no "background" class, and that any loss doesn't propagate to "dead" pixels.
    return x[:, :, :, -self.class_n:]

  @tf.function
  def initialize(self, images):
    state = tf.zeros([tf.shape(images)[0], self.grid_size_y, self.grid_size_x, self.channel_n - 1])  # (bs, 5, 4, n-1)
    images = tf.reshape(images, [-1, self.grid_size_y, self.grid_size_x, 1])  # (bs, 5, 4, 1)
    return tf.concat([images, state], -1)  # (bs, 5, 4, n)


class Node:
  def __init__(self, name, pk_self, pk_bottom, pk_left, pk_right, pk_top, perceive_bias, dmodel_kernel_1, dmodel_bias_1,
               dmodel_kernel_2, dmodel_bias_2, n_val: RawArray, e_val: RawArray, s_val: RawArray, w_val: RawArray,
               own_val: RawArray):
    self.name = name
    self.w_val = w_val
    self.s_val = s_val
    self.e_val = e_val
    self.n_val = n_val
    self.own_val = own_val
    self.state = np.frombuffer(own_val, dtype=np.float32)  # (11,)
    self.pk_self = pk_self  # (1, 1, 15, 40)
    self.pk_bottom = pk_bottom  # (1, 1, 15, 40)
    self.pk_left = pk_left  # (1, 1, 15, 40)
    self.pk_right = pk_right  # (1, 1, 15, 40)
    self.pk_top = pk_top  # (1, 1, 15, 40)
    self.perceive_bias = perceive_bias  # (6, )
    self.dmodel_kernel_1 = dmodel_kernel_1  # (1, 1, 15, 40)
    self.dmodel_bias_1 = dmodel_bias_1  # (6, )
    self.dmodel_kernel_2 = dmodel_kernel_2  # (1, 1, 40, 15)
    self.dmodel_bias_2 = dmodel_bias_2  # (15, )

  @classmethod
  def from_pickle(cls, name: str, filename: str, n_val: RawArray, e_val: RawArray, s_val: RawArray, w_val: RawArray,
                  own_val: RawArray):
    dictionary = PicklePersist.decompress_pickle(filename)
    pk_self = dictionary['pk_self']
    pk_bottom = dictionary['pk_bottom']
    pk_left = dictionary['pk_left']
    pk_right = dictionary['pk_right']
    pk_top = dictionary['pk_top']
    perceive_bias = dictionary['perceive_bias']
    dmodel_kernel_1 = dictionary['dmodel_kernel_1']
    dmodel_bias_1 = dictionary['dmodel_bias_1']
    dmodel_kernel_2 = dictionary['dmodel_kernel_2']
    dmodel_bias_2 = dictionary['dmodel_bias_2']
    return Node(name, pk_self, pk_bottom, pk_left, pk_right, pk_top, perceive_bias, dmodel_kernel_1, dmodel_bias_1,
                dmodel_kernel_2, dmodel_bias_2, n_val, e_val, s_val, w_val, own_val)

  # todo need to make it work (or remove)
  @classmethod
  def from_files(cls, name: str, path: str, n_val: RawArray, e_val: RawArray, s_val: RawArray, w_val: RawArray,
                 own_val: RawArray):
    pk_self = Node.fetch_params_from_file(path + 'pk_self.csv')
    pk_bottom = Node.fetch_params_from_file(path + 'pk_bottom.csv')
    pk_left = Node.fetch_params_from_file(path + 'pk_left.csv')
    pk_right = Node.fetch_params_from_file(path + 'pk_right.csv')
    pk_top = Node.fetch_params_from_file(path + 'pk_top.csv')
    perceive_bias = Node.fetch_params_from_file(path + 'perceive_bias.csv')
    dmodel_kernel_1 = Node.fetch_params_from_file(path + 'dmodel_kernel_1.csv')
    dmodel_bias_1 = Node.fetch_params_from_file(path + 'dmodel_bias_1.csv')
    dmodel_kernel_2 = Node.fetch_params_from_file(path + 'dmodel_kernel_2.csv')
    dmodel_bias_2 = Node.fetch_params_from_file(path + 'dmodel_bias_2.csv')
    return Node(name, pk_self, pk_bottom, pk_left, pk_right, pk_top, perceive_bias, dmodel_kernel_1, dmodel_bias_1,
                dmodel_kernel_2, dmodel_bias_2, n_val, e_val, s_val, w_val, own_val)

  @staticmethod
  def fetch_params_from_file(filename: str):
    with open(filename, newline='') as f:
      reader = csv.reader(f)
      return next(reader)

  @staticmethod
  def sync_update_all(nodes: list[Node]):
    for node in nodes:
      node.forward(output=False)
    for node in nodes:
      node.output()

  def forward(self, output: bool = True):
    n, e, s, w = self.sensors()
    x = self.relu(self.state @ self.pk_self +
                  n @ self.pk_top +
                  e @ self.pk_right +
                  w @ self.pk_left +
                  s @ self.pk_bottom + self.perceive_bias)

    # update net
    x = self.relu(x @ self.dmodel_kernel_1 + self.dmodel_bias_1)  # (40,)
    x = x @ self.dmodel_kernel_2 + self.dmodel_bias_2  # (15,)
    self.state = self.state + x  # (11,)
    if output:
      self.output()

  def run(self, n_steps: int = 30, sleep: bool = True):
    print("start " + self.name)
    for i in range(n_steps):
      self.forward()
      print("update %d %s" % (i, self.name))
      if sleep:
        time.sleep(0.1)
    print("done " + self.name)

  @staticmethod
  def relu(x):
    return (x > 0) * x

  def sensors(self):
    return (np.frombuffer(v, dtype=np.float32) if v is not None else np.zeros(21, dtype=np.float32) for v in
            (self.n_val, self.e_val, self.s_val, self.w_val))

  def output(self):
    self.own_val[:] = self.state.tolist()
