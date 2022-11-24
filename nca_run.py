import ctypes
import random
import time
import numpy as np
from multiprocessing import RawArray

from nca import Node
from nca_training import train_and_pickle


def print_vals():
  for i in range(4):
    print("".join(
      [str(np.argmax(np.frombuffer(vals[i][j], dtype=np.float32)[-3:])) if vals[i][j] is not None else '-' for j in
       range(9)]))
  print('')


if __name__ == '__main__':

  train_and_pickle()

  s0 = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0]]
  s1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0]]
  s2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0]]

  x = s0
  # setup shared arrays
  vals = [[None for _ in range(9)] for _ in range(4)]
  for i in range(4):
    for j in range(9):
      if x[i][j] == 1:
        vals[i][j] = RawArray(ctypes.c_float,
                              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0])

  nodes = []
  for i in range(4):
    for j in range(9):
      if x[i][j] == 1:
        node = Node.from_pickle("%d%d" % (i, j), 'params_set1.pbz2',
                                vals[i - 1][j] if i > 0 else None, vals[i][j + 1] if j < 8 else None,
                                vals[i + 1][j] if i < 3 else None, vals[i][j - 1] if j > 0 else None, vals[i][j])
        nodes.append(node)

  for _ in range(28):
    for j in range(len(nodes)):
      num = random.randint(0, len(nodes) - 1)
      node = nodes[num]
      node.forward()
    print_vals()
    time.sleep(0.1)
