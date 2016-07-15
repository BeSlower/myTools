import sys
import lmdb
import numpy as np
from argparse import ArgumentParser


if 'caffe/python' not in sys.path:
    sys.path.insert(0, 'caffe/python')
from caffe.proto.caffe_pb2 import Datum
from scipy.io import savemat


def main(args):
    datum = Datum()
    data = []
    env = lmdb.open(args.input_lmdb)
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= args.truncate: break
            datum.ParseFromString(value)
            data.append(datum.float_data)
    data = np.squeeze(np.asarray(data))
    np.save(args.output_npy, data)
    savemat('feature.mat', {'feature': data})

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_lmdb')
    parser.add_argument('output_npy')
    parser.add_argument('--truncate', type=int, default=np.inf,
            help="Stop converting the items from the database after this. "
                 "All the items will be converted if not specified.")
    args = parser.parse_args()
    main(args)
