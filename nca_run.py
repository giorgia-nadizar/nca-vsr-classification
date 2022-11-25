import ctypes
import random
import sys
import time
from multiprocessing import RawArray

import numpy as np

from nca import Node
from nca_training import load_shapes_from_file


def print_vals():
  for i in range(4):
    print("".join(
      [str(np.argmax(np.frombuffer(vals[i][j], dtype=np.float32)[-3:])) if vals[i][j] is not None else '-' for j in
       range(9)]))
  print('')


if __name__ == '__main__':
  target_set = 2
  target_shape = 5
  n_steps = 50

  args = sys.argv[1:]
  for arg in args:
    if arg.startswith('set'):
      target_set = int(arg.replace('set=', ''))
    elif arg.startswith('shape'):
      target_set = int(arg.replace('shape=', ''))
    elif arg.startswith('steps'):
      target_set = int(arg.replace('steps=', ''))

  shapes = load_shapes_from_file('shapes/sample_creatures_set' + str(target_set) + '.txt')
  x = shapes[target_shape]
  # setup shared arrays
  width = len(x[0])
  height = len(x)
  vals = [[None for _ in range(width)] for _ in range(height)]
  for i in range(height):
    for j in range(width):
      if x[i][j] == 1:
        vals[i][j] = RawArray(ctypes.c_float,
                              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0])

  nodes = []
  for i in range(height):
    for j in range(width):
      if x[i][j] == 1:
        node = Node.from_pickle("%d%d" % (i, j), 'params_set' + str(target_set) + '.pbz2',
                                vals[i - 1][j] if i > 0 else None, vals[i][j + 1] if j < width - 1 else None,
                                vals[i + 1][j] if i < height - 1 else None, vals[i][j - 1] if j > 0 else None,
                                vals[i][j])
        nodes.append(node)

  for _ in range(n_steps):
    for j in range(len(nodes)):
      num = random.randint(0, len(nodes) - 1)
      node = nodes[num]
      node.forward()
    print_vals()
    time.sleep(0.1)
