
import sys
import zlib
import numpy as np


class CompressedArrayList(object):

    def __init__(self, to_compress, verbose=False):

        before = 0
        after = 0

        # compress data
        self.compressed = []
        self.shapes = []
        self.dtypes = []
        for a in to_compress:
            self.shapes.append(a.shape)
            self.compressed.append(zlib.compress(a))
            self.dtypes.append(a.dtype)

            before += sys.getsizeof(a)
            after += sys.getsizeof(self.compressed[-1])

        if verbose:
            print("%.1f -> %.1f" % (before / 1e6, after / 1e6))

    def __getitem__(self, item):

        if item.__class__ == int:
            return np.fromstring(zlib.decompress(self.compressed[item]), self.dtypes[item]).reshape(self.shapes[item])
        else:
            l = []
            for i in item:
                l.append(np.fromstring(zlib.decompress(self.compressed[i]), self.dtypes[i]).reshape(self.shapes[i]))
            return l

    def __len__(self):
        return len(self.compressed)
