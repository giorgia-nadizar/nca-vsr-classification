import bz2
import _pickle


class PicklePersist:

  @staticmethod
  def compress_pickle(filename, data):
    with bz2.BZ2File(filename + '.pbz2', 'w') as f:
      _pickle.dump(data, f)

  @staticmethod
  def decompress_pickle(filename):
    data = bz2.BZ2File(filename, 'rb')
    return _pickle.load(data)
