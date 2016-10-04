__author__ = 'DanielMinsuKim'


import argparse
import leveldb
import pickle
import caffe

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--bundle', type=int)
args = parser.parse_args()
n_bundle = args.bundle
print("bundle num: " % n_bundle)

db = leveldb.LevelDB('torcs_data')

def save_in_pickle(lst, num):
        folder_name = 'pickle/torcs_data_%d.pickle' % num
        with open(folder_name, 'wb') as handle:
                pickle.dump(lst, handle, protocol=2)
                print('saved to %s' % folder_name)

def convertLeveldb_to_pickle(db, sample = True, num = 100):
    i = 0
    lst = []

    for key, value in db.RangeIter():
        dic = {}
        if sample and i == num:
            break
        if i != 0 and i % n_bundle == 0:
            save_in_pickle(lst, i)
            lst = []
        dic['key'] = key
        dic['value'] = value
        lst.append(dic)
        i += 1

convertLeveldb_to_pickle(db, sample = False)



