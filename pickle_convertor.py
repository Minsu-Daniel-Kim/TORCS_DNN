__author__ = 'DanielMinsuKim'

import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os

path = os.getcwd() + "/raw_pickle"
raw_pickles = [item for item in os.listdir(path) if item.endswith(".pickle")]
# raw_pickles = raw_pickles[:1]


def remove_file(pickle_data):
    delte_file = path + "/" + pickle_data
    os.remove(delte_file)

def read_data_from_pickle(pickle_data):
    print("read raw pickle...")
    with open("raw_pickle/%s"%pickle_data, 'rb') as handle:
        data = pickle.load(handle)
        return data

def save_data_to_pickle(title, data):
    print("save to pickle...")
    with open("processed_pickle/%s"%title, 'wb') as handle:
        print("saving....")
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
        print("saved!")

def datum_to_dict(data):
    datum = caffe_pb2.Datum()
    imgs = []
    labels = []
    print("convert datum to dictionary...")
    for item in data:

        key = str(item['key'])

        datum.ParseFromString(str(item['value']))
        data = caffe.io.datum_to_array(datum)
        image = np.transpose(data, (1,2,0))
        label = list(datum.float_data)

        img = {
            "key": key,
            "image": image

              }
        label = {
            "key" : key,
            "label" : label
        }
        imgs.append(img)
        labels.append(label)

    return [imgs, labels]

def process_pickle(pickle_data):
    raw_pickle = read_data_from_pickle(pickle_data)
    processed_dict = datum_to_dict(raw_pickle)
    remove_file(pickle_data)
    save_data_to_pickle(pickle_data, processed_dict)

for raw_pickle in raw_pickles:
    print(raw_pickle)
    process_pickle(raw_pickle)