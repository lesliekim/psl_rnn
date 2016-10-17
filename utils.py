import os
import pickle
import random

def load_set(datadir, dataname):
    filename = os.path.join(datadir, dataname + '_image.p')
    print('Loading data from: ' + filename)
    f = open(filename, 'rb')
    seqImage = pickle.load(f)
    f.close()

    filenmae = os.path.join(datadir, dataname + '_label.p')
    print('Loading label from: ' + filename)
    f = open(filename, 'rb')
    seqLabel = pickle.load(f)
    f.close()
    return seqImage, seqLabel

class Loader(object):
    def __init__(self, datadir, set_list = [], batch_size = 1, norm_width = 32):
        self.count = 0
        self.batch_size = batch_size
        self.norm_width = norm_width
        print('Data Loader initializing ...')
        self.image = []
        self.label = []
        for dataset in set_list:
            pirnt('Set ' + dataset + '...')
            tmpImage, tmpLabel = load_set(datadir, dataset)
            self.image += tmpImage
            self.label += tmpLabel
        self.train_length = len(self.label)
        self.batch_number = int(np.ceil(np.float(self.train_length) / np.float(batch_size)))

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
            print('x batch seq shape: ' ,np.shpae(x_batch_seq[i]))
            x_batch[i, :step_batch[i], :, 0] = np.transpose(x_batch_seq[i][np.newaxis, :, :], (0, 2, 1))
            print('x batch shape: ', np.shape(x_batch))
        # Creating sparse representation to feed the placeholder
        y_batch = sparse_tuple_from(y_batch_seq)
        tar_len_batch = 0
        for y in y_batch_seq:
            tar_len_batch += len(y)
        return x_batch / 255., y_batch, step_batch, tar_len_batch

    def shuffle(self):
        compact = zip(self.image, self.label)
        random.shuffle(compact)
        self.image, self.label = map(list, zip(*compact))
