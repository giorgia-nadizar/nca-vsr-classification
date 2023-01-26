import _pickle
import bz2
import re
from typing import List

from matplotlib import pyplot as plt
from numpy import median
from pandas import DataFrame


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


class PgfplotsUtils:

  @staticmethod
  def line_plot(filename: str, df: DataFrame, x: str, y: list, groups: List[str] = None):
    def q1(x):
      return x.quantile(0.25)

    def q3(x):
      return x.quantile(0.75)

    vals = dict([(key, [q1, q3, median]) for key in y])

    summary = df.groupby(groups + [x]).agg(vals)
    summary.columns = ["_".join(col) for col in summary.columns.to_flat_index()]
    summary.reset_index(inplace=True)

    key_df = df.drop_duplicates(subset=groups)

    for i in range(len(key_df)):
      tmp = summary
      current_filename = filename
      for key in groups:
        tmp = tmp[tmp[key] == key_df[key].iloc[i]]
        current_filename += f"_{key_df[key].iloc[i]}"
      tmp.to_csv(f"{current_filename}.txt", sep="\t", index=False)

  @staticmethod
  def box_plot(filename: str, df: DataFrame, x: str, y: str, groups: List[str] = None):
    if groups is None or len(groups) == 0:
      PgfplotsUtils.__box_plot(df, x, y, filename)

    else:
      key_df = df.drop_duplicates(subset=groups)

      for i in range(len(key_df)):
        tmp = df
        current_filename = filename
        for key in groups:
          tmp = tmp[tmp[key] == key_df[key].iloc[i]]
          current_filename += f"_{key_df[key].iloc[i]}"

        PgfplotsUtils.__box_plot(tmp, x, y, current_filename)

  @staticmethod
  def __box_plot(df: DataFrame, x: str, y: str, filename: str):
    plt.figure(visible=False)
    data = []
    for xi in df[x].unique():
      data.append([k for k in df[df[x] == xi][y] if str(k) != "nan"])

    bp = plt.boxplot(data, showmeans=False)

    minimums = [round(item.get_ydata()[0], 1) for item in bp['caps']][::2]
    q1 = [round(min(item.get_ydata()), 1) for item in bp['boxes']]
    medians = [item.get_ydata()[0] for item in bp['medians']]
    q3 = [round(max(item.get_ydata()), 1) for item in bp['boxes']]
    maximums = [round(item.get_ydata()[0], 1) for item in bp['caps']][1::2]

    rows = [df[x].unique().tolist(), minimums, q1, medians, q3, maximums]

    with open(f"{filename}.txt", "w") as bp_file:
      for row in rows:
        bp_file.write("\t".join(map(str, row)) + "\n")
