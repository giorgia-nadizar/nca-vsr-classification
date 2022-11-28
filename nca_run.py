import ctypes
import sys
import time
from multiprocessing import RawArray

import numpy as np

from nca import Node
from nca_training import load_shapes_from_file, parse_shape


def print_vals(vals, width: int, height: int, n_classes: int) -> None:
  for i in range(height):
    print("".join(
      [str(np.argmax(np.frombuffer(vals[i][j], dtype=np.float32)[-n_classes:])) if vals[i][j] is not None else '-'
       for j in range(width)]))


def main(sleep: bool, display_transient: bool, target_set: int, target_shape: str, n_steps: int, deterministic: bool):
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
    if deterministic:
      Node.sync_update_all(nodes)
    else:
      Node.stochastic_update(nodes)
    if display_transient or n == n_steps - 1:
      print_vals(vals, width, height, len(shapes))
    if display_transient:
      print('')
    if sleep:
      time.sleep(0.1)


if __name__ == '__main__':
  m_sleep = False
  m_display_transient = False
  m_target_set = 1
  m_target_shape = '0'
  m_n_steps = 50
  m_deterministic = True

  args = sys.argv[1:]
  for arg in args:
    if arg.startswith('set'):
      m_target_set = int(arg.replace('set=', ''))
    elif arg.startswith('shape'):
      m_target_shape = arg.replace('shape=', '')
    elif arg.startswith('steps'):
      m_n_steps = int(arg.replace('steps=', ''))
    elif arg.startswith('sleep'):
      m_sleep = arg.replace('sleep=', '').lower().startswith('t')
    elif arg.startswith('display_transient'):
      m_display_transient = arg.replace('display_transient=', '').lower().startswith('t')
    elif arg.startswith('deterministic'):
      m_deterministic = arg.replace('deterministic=', '').lower().startswith('t')

  main(m_sleep, m_display_transient, m_target_set, m_target_shape, m_n_steps, m_deterministic)
