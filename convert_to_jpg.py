__author__ = 'DanielMinsuKim'


import scipy.misc
import os
from PIL import Image
import cv2
import pickle
import json

LOGDIR = 'driving_dataset2'

def read_data_from_processed_pickle(pickle_data):
    print("read processed pickle...")
    with open("../processed_pickle/%s" % pickle_data, 'rb') as handle:
        # data = pickle.load(handle,encoding='latin1')
        data = pickle.load(handle)
        return data

# dataset = read_data_from_processed_pickle("")
path = os.getcwd() + "/../processed_pickle"
print("path: %s" % path)
processed_pickles = [item for item in os.listdir(path) if item.endswith(".pickle")]
processed_pickles = processed_pickles[:1]

dataset = []
for item in processed_pickles:

    dataset.extend(read_data_from_processed_pickle(item))
if not os.path.exists(LOGDIR):
    print("make with folder")
    os.makedirs(LOGDIR)


i = 0

data_label = {}
le = len(dataset[0])

while le > 9980:

    file_name = "%d.jpg" % i
    checkpoint_path = os.path.join(LOGDIR, file_name)
    print(checkpoint_path)
    # image_array['image']
    # tmp = dataset[0][i]
    # print(tmp['image'])

    im = Image.fromarray(dataset[0][i]['image'])
    im.save(checkpoint_path)
    data_label[file_name] = 1

    data_label[file_name] = dataset[1][i]['label'][0]

    i += 1
    le -= 1

with open(os.path.join(LOGDIR, 'data.txt'), 'w') as outfile:
    json.dump(data_label, outfile)


# for image_array in dataset[0]:
#     file_name = "%d.jpg" % i
#     checkpoint_path = os.path.join(LOGDIR, file_name)
#     image_array['image']
#     im = Image.fromarray(image_array['image'])
#     im.save(checkpoint_path)
#
#     data_label[file_name] = 1
#
#
#     # with open("Output.txt", "w") as text_file:
#     # text_file.write("Purchase Amount: %s" % TotalAmount)
#
#     i += 1

# print(dataset[0][0])
# image_array = dataset[0][0]
# scipy.misc.imsave("test.jpg", image_array)
# scipy.misc.toimage(image_array).save('outfile.jpg')

# for image_array in dataset:
#
#     checkpoint_path = os.path.join(LOGDIR, "image%d" % i)
#     scipy.misc.imsave(checkpoint_path, image_array)
#
#     # print("Model saved in file: %s" % filename)
#
#     i += 1