import os
import pickle
import random
import math
import numpy as np
import classify_param as param
import cv2 as cv


'''
Data Loader
'''
def load_set(datadir, dataname):
    filename = os.path.join(datadir, dataname + '_image.p')
    print('Loading data from: ' + filename)
    f = open(filename, 'rb')
    images = pickle.load(f)
    f.close()

    filename = os.path.join(datadir, dataname + '_label.p')
    print('Loading label from: ' + filename)
    f = open(filename, 'rb')
    labels = pickle.load(f)
    f.close()
    return images, labels

def sparse_array(sequences):
    classes = param.num_classes
    indices = np.zeros((len(sequences), classes), dtype=np.int64)

    for i,seq in enumerate(sequences):
        indices[i][int(seq)] = 1

    return indices

def pad_array(sequences, width):
    indices = []
    for i,seq in enumerate(sequences):
        ans = [0] * width
        ans[:len(seq)] = seq[:]
        indices.append(ans)

    indices = np.asarray(indices, dtype=np.int64)
    return indices

def get_row(sparse_tuple, row, dtype=np.int32):
    optlist = []
    cnt = 0
    for pos in sparse_tuple[0]:
        if pos[0] == row:
            optlist.append(sparse_tuple[1][cnt])
        cnt += 1
    return optlist

class Loader(object):
    def __init__(self, datadir, set_list = [], batch_size = 1): 
        self.count = 0
        self.batch_size = batch_size
        print('Data Loader initializing ...')
        self.image = []
        self.label = []
        for dataset in set_list:
            print('Set ' + dataset + '...')
            tmpImage, tmpLabel = load_set(datadir, dataset)
            self.image += tmpImage
            self.label += tmpLabel

        self.train_length = len(self.image)
        self.batch_number = int(np.ceil(np.float(self.train_length) / np.float(batch_size)))
        

    def next_batch(self):
        if self.count + self.batch_size <= self.train_length:
            x_batch_seq = self.image[self.count : self.count + self.batch_size]
            y_batch_seq = self.label[self.count : self.count + self.batch_size]
            self.count += self.batch_size
        else:
            left_num = self.count + self.batch_size - self.train_length
            x_batch_seq = self.image[self.count :] + self.image[0 : left_num]
            y_batch_seq = self.label[self.count :] + self.label[0 : left_num]
            self.count = 0

        x_batch = np.reshape(np.asarray(x_batch_seq), (self.batch_size, param.resize_height, param.resize_width, 1))
        x_batch = x_batch / 255.
        y_batch = sparse_array(y_batch_seq)
        
        return x_batch, y_batch

    def shuffle(self):
        compact = zip(self.image, self.label)
        random.shuffle(compact)
        self.image, self.label = map(list, zip(*compact))

