import ctypes
import sys
import time
from multiprocessing import RawArray
from typing import List, Tuple
import nltk

import numpy as np

from nca import Node
from nca_training import load_shapes_from_file, parse_shape


def compute_edit_distance_between_shapes(shape1: List[List[int]], shape2: List[List[int]]):
  h1, l1 = len(shape1), len(shape1[0])
  h2, l2 = len(shape2), len(shape2[0])
  total_h = h1 + 2 * (h2 - 1)
  total_l = l1 + 2 * (l2 - 1)
  padded_shape1 = []
  for _ in range(h2 - 1):
    padded_shape1.append([0] * total_l)
  for row in shape1:
    padded_shape1.append([0] * (l2 - 1) + row + [0] * (l2 - 1))
  for _ in range(h2 - 1):
    padded_shape1.append([0] * total_l)

  padded_shape1_string = shape_to_string(padded_shape1, hide_extra_zeros=False)

  min_edit_distance = np.inf
  # shift shape 2 in all grid positions
  for y_off in range(total_h - h2 + 1):
    for x_off in range(total_l - l2 + 1):
      # compute the shape
      shifted_shape2 = []
      for _ in range(y_off):
        shifted_shape2.append([0] * total_l)
      for row in shape2:
        shifted_shape2.append([0] * x_off + row + [0] * (total_l - x_off - l2))
      for _ in range(total_h - h2 - y_off):
        shifted_shape2.append([0] * total_l)
      shifted_shape2_string = shape_to_string(shifted_shape2, hide_extra_zeros=False)
      edit_distance = nltk.edit_distance(padded_shape1_string, shifted_shape2_string)
      if edit_distance < min_edit_distance:
        min_edit_distance = edit_distance
  return min_edit_distance


def shape_to_string(shape: List[List[int]], hide_extra_zeros: bool = True) -> str:
  mi = min([row.index(1) for row in shape if 1 in row]) if hide_extra_zeros else 0
  ma = max([len(row) - 1 - row[::-1].index(1) for row in shape if 1 in row]) if hide_extra_zeros else len(shape[0]) - 1
  strings = []
  for row in shape:
    if (hide_extra_zeros and 1 in row) or not hide_extra_zeros:
      strings.append(''.join(str(row[k]) for k in range(mi, ma + 1)))
  return '-'.join(strings)


def classification_accuracy(ground_truth_id: int, classification: str) -> Tuple[float, int]:
  prediction_classes = [int(p.split(';')[2]) for p in classification.split('-')]
  accuracy = prediction_classes.count(ground_truth_id) / len(prediction_classes)
  majority_vote = max(set(prediction_classes), key=prediction_classes.count)
  return accuracy, majority_vote


def string_vals(vals, width: int, height: int, n_classes: int, pretty_print: bool = True, inline: bool = False,
                reversed_y_axis=False) -> str:
  values = []
  if pretty_print:
    for h in range(height):
      i = h if reversed_y_axis else height - 1 - h
      values.append("".join(
        [f'({np.argmax(np.frombuffer(vals[i][j], dtype=np.float32)[-n_classes:]):02d})'
         if vals[i][j] is not None else '    ' for j in range(width)]))
  else:
    for h in range(height):
      for j in range(width):
        i = h if reversed_y_axis else height - 1 - h
        if vals[i][j] is not None:
          values.append(f'{j};{h};{np.argmax(np.frombuffer(vals[i][j], dtype=np.float32)[-n_classes:])}')
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


def correct_shapes_classification_csv(n_steps: int = 101, n_snapshots: int = 101, n_extra_channels: int = 10,
                                      deterministic: bool = True,
                                      accuracy_column: bool = True):
  target_sets = range(1, 5)
  with open('classifications/classification.csv', 'w') as f:
    f.write('target_set,shape_id,readable_shape,step,classification')
    if accuracy_column:
      f.write(',accuracy,majority_vote')
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
            f.write(f'{target_set},{shape_id},{shape_to_string(shape)},{n},{classification_string}')
            if accuracy_column:
              accuracy, majority_vote = classification_accuracy(shape_id, classification_string)
              f.write(f',{accuracy},{majority_vote}')
            f.write('\n')


def mismatched_shapes_classification_csv(shapes_set: int, nca_set: int, n_steps: int = 101, n_snapshots: int = 101,
                                         n_extra_channels: int = 10, deterministic: bool = True):
  with open('classifications/mismatched_classification.csv', 'w') as f:
    f.write('shapes_set,shape_id,target_set,closest_shape_id,edit_distance,readable_shape,step,classification,'
            'majority_vote\n')
    shapes = load_shapes_from_file('shapes/sample_creatures_set' + str(shapes_set) + '.txt')
    nca_shapes = load_shapes_from_file('shapes/sample_creatures_set' + str(nca_set) + '.txt')
    n_classes = len(nca_shapes)
    step = n_steps // n_snapshots
    for shape_id, shape in enumerate(shapes):
      ids_distances = dict(
        [(idx, compute_edit_distance_between_shapes(nca_shapes[idx], shape)) for idx in range(len(nca_shapes))])
      closest_shape_id = min(ids_distances, key=ids_distances.get)
      edit_distance = ids_distances[closest_shape_id]

      width = len(shape[0])
      height = len(shape)
      vals, nodes = setup_nca(nca_shapes, shape, n_extra_channels, nca_set)
      for n in range(n_steps):
        if deterministic:
          Node.sync_update_all(nodes)
        else:
          Node.stochastic_update(nodes)
        if n % step == 0:
          classification_string = string_vals(vals, width, height, n_classes, pretty_print=False, inline=True)
          _, majority_vote = classification_accuracy(shape_id, classification_string)
          f.write(f'{shapes_set},{shape_id},{nca_set},{closest_shape_id},{edit_distance},{shape_to_string(shape)},{n},'
                  f'{classification_string},{majority_vote}\n')


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
      print(string_vals(vals, width, height, n_classes, pretty_print, reversed_y_axis=True))
    if display_transient:
      print('')
    if sleep:
      time.sleep(0.1)


if __name__ == '__main__':
  m_sleep = False
  m_display_transient = True
  m_target_set = 1
  m_target_shape = '0'
  m_n_steps = 25
  m_deterministic = True
  m_pretty_print = False

  args = sys.argv[1:]
  for arg in args:
    if arg.startswith('csv_mismatch'):
      mismatched_shapes_classification_csv(3, 1)
      exit(0)
    if arg.startswith('csv'):
      correct_shapes_classification_csv()
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
