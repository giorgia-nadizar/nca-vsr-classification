import bz2
import _pickle
import re


class PicklePersist:

  @staticmethod
  def compress_pickle(filename, data):
    with bz2.BZ2File(filename + '.pbz2', 'w') as f:
      _pickle.dump(data, f)

  @staticmethod
  def decompress_pickle(filename):
    data = bz2.BZ2File(filename, 'rb')
    return _pickle.load(data)


class ShapeUtils:

  @staticmethod
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

  @staticmethod
  def load_shapes_from_file(filename: str):
    shapes = []
    lines = open(filename, 'r').readlines()
    counter = 0
    for line in lines:
      shape = ShapeUtils.parse_shape(line)
      counter += 1
      if shape:
        shapes.append(shape)
    return shapes
