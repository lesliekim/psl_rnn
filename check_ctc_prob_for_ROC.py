import os
import sys
import cv2 as cv
import utils
import seg_utils
import numpy as np

epoch = int(sys.argv[1])

padding = 0.257449 # you should change its value every time
pooling_size = 1 # if RNN, set this parameter to 1
data_dir = '/home/jia/psl/tf_rnn/psl_data/father/seg_synthesis_traindata'
ctc_dir = '/home/jia/psl/tf_rnn/psl_rnn/model/syntheticFatherRnnOnly/ctc'
output_dir = os.path.join(ctc_dir, 'roc')

assert os.path.isdir(data_dir)
assert os.path.isdir(ctc_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# make sure your image data order matchs your ctc file order
test_data = ['data_4','data_8','data_12','data_16']
test_loader = utils.Loader(data_dir, test_data, 32)

image_count = 1 # make sure its initial number is the same with test_bi_rnn.py!!!
batch_label = test_loader.label
print np.asarray(batch_label).shape
match_seq = []
match_line_number = 0
truth_line_number = 0
for label in batch_label:
    if image_count % 100 == 0:
        print('finish {}images'.format(image_count))

    ctc_file_name = 'rnn_prob_ep{}_{}.txt'.format(epoch, image_count)
    image_count += 1
    # read ctc probability file
    ctc_prob = utils.read_probfile(os.path.join(ctc_dir, ctc_file_name))
    # move padding
    print 'ctc_prob len: ' + str(len(ctc_prob))
    prob_line = utils.move_padding(ctc_prob, padding)
    # match
    batch_match_seq, batch_match_line_number = \
            seg_utils.match(label, prob_line, 4)

    match_seq += batch_match_seq
    match_line_number += batch_match_line_number
    truth_line_number += label.count(1)

print match_line_number, truth_line_number
roc_line = seg_utils.single_roc_line(match_seq, match_line_number, truth_line_number)
