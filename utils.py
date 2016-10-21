import os
import pickle
import random
import numpy as np

def load_set(datadir, dataname):
    filename = os.path.join(datadir, dataname + '_image.p')
    print('Loading data from: ' + filename)
    f = open(filename, 'rb')
    seqImage = pickle.load(f)
    f.close()

    filename = os.path.join(datadir, dataname + '_label.p')
    print('Loading label from: ' + filename)
    f = open(filename, 'rb')
    seqLabel = pickle.load(f)
    f.close()
    return seqImage, seqLabel
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for i, seq in enumerate(sequences):
        #print('seq len: ',len(seq))
        #print(seq)
        indices.extend(zip([i] * len(seq), xrange(len(seq))))
        values.extend(seq)
        #print('values len: ', len(values))
    
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape
def get_row(sparse_tuple, row, dtype=np.int32):
    optlist = []
    cnt = 0
    for pos in sparse_tuple[0]:
        if pos[0] == row:
            optlist.append(sparse_tuple[1][cnt])
        cnt += 1
    return optlist

class Loader(object):
    def __init__(self, datadir, set_list = [], batch_size = 1, norm_width = 32):
        self.count = 0
        self.batch_size = batch_size
        self.norm_width = norm_width
        print('Data Loader initializing ...')
        self.image = []
        self.label = []
        for dataset in set_list:
            print('Set ' + dataset + '...')
            tmpImage, tmpLabel = load_set(datadir, dataset)
            self.image += tmpImage
            self.label += tmpLabel
        self.train_length = len(self.label)
        self.batch_number = int(np.ceil(np.float(self.train_length) / np.float(batch_size)))
        
        self.target_len = 0
        for target in self.label:
            self.target_len += len(target)

    def next_batch(self):
        if self.count + self.batch_size < self.train_length:
            x_batch_seq = self.image[self.count : self.count + self.batch_size]
            y_batch_seq = self.label[self.count : self.count + self.batch_size]
            self.count += self.batch_size
        else:
            x_batch_seq = self.image[self.count :]
            y_batch_seq = self.label[self.count :]
            self.count = 0

        step_batch = np.zeros(shape=[len(x_batch_seq)], dtype='int64')
        for i in xrange(len(step_batch)):
            step_batch[i] = np.shape(x_batch_seq[i])[1]
        x_batch = np.zeros(shape=[len(step_batch), np.max(step_batch), self.norm_width, 1])
        for i in xrange(len(step_batch)):
            x_batch[i, :step_batch[i], :, 0] = np.transpose(x_batch_seq[i][np.newaxis, :, :], (0, 2, 1))

        # Creating sparse representation to feed the placeholder
        #print(y_batch_seq)
        y_batch = sparse_tuple_from(y_batch_seq)
        tar_len_batch = 0
        for y in y_batch_seq:
            tar_len_batch += len(y)
        return x_batch / 255., y_batch, step_batch, tar_len_batch

    def shuffle(self):
        compact = zip(self.image, self.label)
        random.shuffle(compact)
        self.image, self.label = map(list, zip(*compact))
'''
train_loader = Loader('../psl_data/244Images/traindata', ['inputfile_0'], 16, 32)
x_train, y_train, step_batch, tar_len_batch = train_loader.next_batch()
print('x_train: ', np.shape(x_train))
print('y_train: ', np.shape(y_train))
print('step_batch: ', np.shape(step_batch))
print('tar_len_batch: ', np.shape(tar_len_batch))
'''
