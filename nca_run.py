import ctypes
import random
import sys
import time
from multiprocessing import RawArray

import numpy as np

from nca import Node
from nca_training import load_shapes_from_file, parse_shape


def print_vals(width: int, height: int, n_classes: int) -> None:
  for i in range(height):
    print("".join(
      [str(np.argmax(np.frombuffer(vals[i][j], dtype=np.float32)[-n_classes:])) if vals[i][j] is not None else '-'
       for j in range(width)]))


if __name__ == '__main__':
  sleep = False
  display_transient = False
  target_set = 1
  target_shape = '0'
  n_steps = 50

  args = sys.argv[1:]
  for arg in args:
    if arg.startswith('set'):
      target_set = int(arg.replace('set=', ''))
    elif arg.startswith('shape'):
      target_shape = arg.replace('shape=', '')
    elif arg.startswith('steps'):
      n_steps = int(arg.replace('steps=', ''))
    elif arg.startswith('sleep'):
      sleep = arg.replace('sleep=', '').lower().startswith('t')
    elif arg.startswith('display_transient'):
      display_transient = arg.replace('display_transient=', '').lower().startswith('t')

  shapes = load_shapes_from_file('shapes/sample_creatures_set' + str(target_set) + '.txt')
  x = shapes[int(target_shape)] if target_shape.isnumeric() else parse_shape(target_shape)

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
        node = Node.from_pickle("%d%d" % (i, j), 'parameters/params_set' + str(target_set) + '.pbz2',
                                vals[i - 1][j] if i > 0 else None, vals[i][j + 1] if j < width - 1 else None,
                                vals[i + 1][j] if i < height - 1 else None, vals[i][j - 1] if j > 0 else None,
                                vals[i][j])
        nodes.append(node)

  for n in range(n_steps):
    for j in range(len(nodes)):
      num = random.randint(0, len(nodes) - 1)
      node = nodes[num]
      node.forward()
    if display_transient or n == n_steps - 1:
      print_vals(width, height, len(shapes))
    if display_transient:
      print('')
    if sleep:
      time.sleep(0.1)
