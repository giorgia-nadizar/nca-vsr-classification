import ctypes
import sys
import time
from multiprocessing import RawArray
from typing import List, Tuple

import numpy as np

from nca import Node
from nca_training import load_shapes_from_file, parse_shape


def shape_to_string(shape: List[List[int]]):
  mi = min([row.index(1) for row in shape if 1 in row])
  ma = max([len(row) - 1 - row[::-1].index(1) for row in shape if 1 in row])
  strings = []
  for row in shape:
    if 1 in row:
      strings.append(''.join(str(row[k]) for k in range(mi, ma + 1)))
  return '-'.join(strings)


def classification_accuracy(ground_truth_id: int, classification: str) -> Tuple[float, int]:
  prediction_classes = [int(p.split(',')[2]) for p in classification.split('-')]
  accuracy = prediction_classes.count(ground_truth_id) / len(prediction_classes)
  majority_vote = max(set(prediction_classes), key=prediction_classes.count)
  return accuracy, majority_vote


def string_vals(vals, width: int, height: int, n_classes: int, pretty_print: bool = True, inline: bool = False) -> str:
  if pretty_print:
    for i in range(height):
      return "".join(
        [f'({np.argmax(np.frombuffer(vals[i][j], dtype=np.float32)[-n_classes:]):02d})'
         if vals[i][j] is not None else '    ' for j in range(width)])
  else:
    values = []
    for i in range(height):
      for j in range(width):
        if vals[i][j] is not None:
          values.append(f'{j},{i},{np.argmax(np.frombuffer(vals[i][j], dtype=np.float32)[-n_classes:])}')
    if inline:
      return '-'.join(values)
    else:
      return '\n'.join(values)


def setup_nca(shapes, x, n_extra_channels: int, target_set: int):
  n_classes = len(shapes)
  width = len(x[0])
  height = len(x)
  # setup shared arrays
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
  return vals, nodes


def main_to_csv(n_steps: int = 51, n_snapshots: int = 5, n_extra_channels: int = 10, deterministic: bool = True,
                accuracy_column: bool = True):
  target_sets = range(1, 5)
  with open('classifications/classification.txt', 'w') as f:
    f.write('target_set;shape_id;readable_shape;step;classification[x,y,c]')
    if accuracy_column:
      f.write(';accuracy;majority_vote')
    f.write('\n')
    for target_set in target_sets:
      shapes = load_shapes_from_file('shapes/sample_creatures_set' + str(target_set) + '.txt')
      n_classes = len(shapes)
      step = n_steps // n_snapshots
      for shape_id, shape in enumerate(shapes):
        width = len(shape[0])
        height = len(shape)
        vals, nodes = setup_nca(shapes, shape, n_extra_channels, target_set)
        for n in range(n_steps):
          if deterministic:
            Node.sync_update_all(nodes)
          else:
            Node.stochastic_update(nodes)
          if n % step == 0:
            classification_string = string_vals(vals, width, height, n_classes, pretty_print=False, inline=True)
            f.write(f'{target_set};{shape_id};{shape_to_string(shape)};{n};{classification_string}')
            if accuracy_column:
              accuracy, majority_vote = classification_accuracy(shape_id, classification_string)
              f.write(f';{accuracy};{majority_vote}')
            f.write('\n')


def main(sleep: bool, display_transient: bool, target_set: int, target_shape: str, n_steps: int, deterministic: bool,
         pretty_print: bool, n_extra_channels: int = 10):
  shapes = load_shapes_from_file('shapes/sample_creatures_set' + str(target_set) + '.txt')
  if target_shape.isnumeric() and int(target_shape) < len(shapes):
    x = shapes[int(target_shape)]
  else:
    x = parse_shape(target_shape)
  n_classes = len(shapes)
  width = len(x[0])
  height = len(x)
  vals, nodes = setup_nca(shapes, x, n_extra_channels, target_set)

  for n in range(n_steps):
    if deterministic:
      Node.sync_update_all(nodes)
    else:
      Node.stochastic_update(nodes)
    if display_transient or n == n_steps - 1:
      print(string_vals(vals, width, height, n_classes, pretty_print))
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
    if arg.startswith('csv'):
      main_to_csv()
      exit(0)
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
