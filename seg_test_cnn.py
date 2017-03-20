import seg_utils as utils
import tensorflow as tf
from model import SegCnnNet as Net
import os
import sys
import numpy as np
import math

# Training configs
model_dir = sys.argv[1]
epoch = int(sys.argv[2])

pooling_size = 1

model_file = os.path.join(model_dir, 'model-{}'.format(epoch))
batch_size = 8
output_dir = '../psl_data/father/word_space_segment_output_2'#'../psl_data/father/ROC_test'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Loading the data
test_loader = utils.Loader('../psl_data/seg_cnn/traindata_total_random',['data_3'],batch_size)

# parameter for roc
test_roc = False
truth_seqs = []
forecast_seqs = []

# paramter for over/under segmentation rate
test_over_under = True
softmaxs = []
targets = []

# test
with tf.Graph().as_default():
    sess = tf.Session()
    model = Net(is_test=True)
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    sess.run(init)

    print "Loading pb file from: {}".format(model_file)
    saver.restore(sess, model_file)

    for i in xrange(test_loader.batch_number):
        #print "Processing batch {} / {}".format(i, test_loader.batch_number)
        # todo: add evaluation result saver
        batch_x_test, batch_y_test, batch_steps_test, _ = test_loader.next_batch()
        '''
        # ROC
        softmax_outputs, pooled_targets =\
            sess.run([model.softmax_outputs, model.pooled_targets], feed_dict={
            model.inputs: batch_x_test,
            model.targets: batch_y_test,
        })
        
        truth_seqs.append(pooled_targets[:,:,1].tolist())
        forecast_seqs.append(softmax_outputs[:,:,1].tolist())
        '''
        
        '''
        # save output
        argmax_outputs =\
            sess.run([model.argmax_outputs], feed_dict={
            model.inputs: batch_x_test,
        })
        utils.crop_and_save(batch_x_test, argmax_outputs[0], pooling_size, 
                               image_normalize=255.0,is_save=True, output_dir=output_dir,
                                batch_number=i)
        '''
        
        '''
        # word segment
        argmax_outputs =\
            sess.run([model.argmax_outputs], feed_dict={
            model.inputs: batch_x_test,
        })
        utils.word_segmentation(batch_x_test, argmax_outputs[0], pooling_size, 
                image_normalize=255.0,is_save=True, output_dir=output_dir, batch_number=i)
        # save line groundtruth
        for j, line in enumerate(batch_y_test):
            txt_name = "batch_{}_line_{}.txt".format(i, j)
            with open(os.path.join(output_dir, txt_name), 'w') as wf:
                for char in line:
                    wf.write(chr(char))
        '''

        # over / under rate
        argmax_targets, softmax_outputs = sess.run([model.argmax_targets, model.softmax_outputs], 
                feed_dict={
                    model.inputs: batch_x_test,
                    model.targets: batch_y_test,
                    })
        softmaxs.append(softmax_outputs)
        targets.append(argmax_targets)
        print 'batch_y_test: ', batch_y_test

    if test_roc:
        # ROC
        total_roc = utils.ROC(truth_seqs, forecast_seqs)

        # save
        with open(os.path.join(output_dir, 'roc_temp.csv'), 'w') as f:
            for j in xrange(len(total_roc)):
                f.write(str(total_roc[j][0]))
                f.write(',')
                f.write(str(total_roc[j][1]))
                f.write(',')
                f.write(str(total_roc[j][2]))
                f.write('\n')

    if test_over_under:
        utils.over_and_under_rate(softmaxs, targets)
