import ctypes
import sys
import time
from multiprocessing import RawArray

import numpy as np

from nca import Node
from nca_training import load_shapes_from_file, parse_shape


def print_vals(vals, width: int, height: int, n_classes: int, pretty_print: bool = True) -> None:
  if pretty_print:
    for i in range(height):
      print("".join(
        [f'({np.argmax(np.frombuffer(vals[i][j], dtype=np.float32)[-n_classes:]):02d})'
         if vals[i][j] is not None else '    ' for j in range(width)]))
  else:
    for i in range(height):
      for j in range(width):
        if vals[i][j] is not None:
          print(f'{j},{i},{np.argmax(np.frombuffer(vals[i][j], dtype=np.float32)[-n_classes:])}')


def main(sleep: bool, display_transient: bool, target_set: int, target_shape: str, n_steps: int, deterministic: bool,
         pretty_print: bool, n_extra_channels: int = 10):
  shapes = load_shapes_from_file('shapes/sample_creatures_set' + str(target_set) + '.txt')
  x = shapes[int(target_shape)] if target_shape.isnumeric() else parse_shape(target_shape)
  n_classes = len(shapes)

  # setup shared arrays
  width = len(x[0])
  height = len(x)
  vals = [[None for _ in range(width)] for _ in range(height)]
  for i in range(height):
    for j in range(width):
      if x[i][j] == 1:
        values = [0.0 for _ in range(n_classes + n_extra_channels)]
        values[0] = 1.0
        vals[i][j] = RawArray(ctypes.c_float, values)

  nodes = []
  for i in range(height):
    for j in range(width):
      if x[i][j] == 1:
        node = Node.from_pickle("%d%d" % (i, j), 'parameters/params_set' + str(target_set) + '.pbz2',
                                vals[i - 1][j] if i > 0 else None, vals[i][j + 1] if j < width - 1 else None,
                                vals[i + 1][j] if i < height - 1 else None, vals[i][j - 1] if j > 0 else None,
                                vals[i][j], n_classes=n_classes)
        nodes.append(node)

  for n in range(n_steps):
    if deterministic:
      Node.sync_update_all(nodes)
    else:
      Node.stochastic_update(nodes)
    if display_transient or n == n_steps - 1:
      print_vals(vals, width, height, n_classes, pretty_print)
    if display_transient:
      print('')
    if sleep:
      time.sleep(0.1)


if __name__ == '__main__':
  m_sleep = False
  m_display_transient = True
  m_target_set = 1
  m_target_shape = '1'
  m_n_steps = 10
  m_deterministic = True
  m_pretty_print = True

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
    elif arg.startswith('pretty_print'):
      m_pretty_print = arg.replace('pretty_print=', '').lower().startswith('t')

  main(m_sleep, m_display_transient, m_target_set, m_target_shape, m_n_steps, m_deterministic, m_pretty_print)
