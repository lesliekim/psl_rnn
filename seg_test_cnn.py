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

pooling_size = 4

model_file = os.path.join(model_dir, 'model-{}'.format(epoch))
batch_size = 8
output_dir = '../psl_data/father/ROC_test'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Loading the data
test_loader = utils.Loader('../psl_data/father/seg_synthesis_traindata',['data_4'],batch_size)

# parameter for roc
truth_seqs = []
forecast_seqs = []

# test
with tf.Graph().as_default():
    sess = tf.Session()
    model = Net(batch_size, is_test=True)
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    sess.run(init)

    print "Loading pb file from: {}".format(model_file)
    saver.restore(sess, model_file)

    for i in xrange(test_loader.batch_number):
        #print "Processing batch {} / {}".format(i, test_loader.batch_number)
        # todo: add evaluation result saver
        batch_x_test, batch_y_test, batch_steps_test, _ = test_loader.next_batch()

        softmax_outputs, pooled_targets =\
            sess.run([model.softmax_outputs, model.pooled_targets], feed_dict={
            model.inputs: batch_x_test,
            model.targets: batch_y_test,
        })
        # save output
        #utils.crop_and_save(batch_x_test, argmax_outputs[0], output_dir, i, pooling_size)
        
        # ROC
        truth_seqs.append(pooled_targets[:,:,1].tolist())
        forecast_seqs.append(softmax_outputs[:,:,1].tolist())

    # ROC
    total_roc = utils.ROC(truth_seqs, forecast_seqs)

    # save
    with open(os.path.join(output_dir, 'roc.csv'), 'w') as f:
        for j in xrange(len(total_roc)):
            f.write(str(total_roc[j][0]))
            f.write(',')
            f.write(str(total_roc[j][1]))
            f.write(',')
            f.write(str(total_roc[j][2]))
            f.write('\n')
