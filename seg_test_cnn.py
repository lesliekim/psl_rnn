import seg_utils as utils
import tensorflow as tf
from model import SegCnnNet as Net
import os
import sys
import numpy as np
import math

np.set_printoptions(threshold='nan')
# Training configs
model_dir = sys.argv[1]
epoch = int(sys.argv[2])

pooling_size = 1

model_file = os.path.join(model_dir, 'model-{}'.format(epoch))
batch_size = 8
output_dir = '../psl_data/seg_cnn/father4_base_outputs'#'../psl_data/father/ROC_test'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Loading the data
test_loader = utils.Loader('../psl_data/seg_cnn/traindata_father_4',['data_{}'.format(i) for i in xrange(1)],batch_size)

# parameter for roc
test_roc = False
truth_seqs = []
forecast_seqs = []

# parameter for over/under segmentation rate
test_over_under = True
softmaxs = []
targets = []

# parameter for save outputs
save_output = False

# normalize func
def norm(batch_feature):
    for k in xrange(batch_feature.shape[0]):
        amin = np.amin(batch_feature[k])
        amax = np.amax(batch_feature[k])
        for h in xrange(batch_feature.shape[1]):
            for w in xrange(batch_feature.shape[2]):
                for fd in xrange(batch_feature.shape[3]):
                    batch_feature[k][h][w][fd] = 1.0 - float(batch_feature[k][h][w][fd] - amin) / (amax - amin)
    return batch_feature

acc_value = 0.0
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
        batch_feature = np.reshape(np.sum(batch_x_test, axis=1),[batch_x_test.shape[0], 1, -1, 1])
        batch_feature = norm(batch_feature)
         
        # ROC
        if test_roc:
            softmax_outputs, pooled_targets =\
                sess.run([model.softmax_outputs, model.pooled_targets], feed_dict={
                model.inputs: batch_x_test,
                model.targets: batch_y_test,
                model.features: batch_feature,
            })
            
            truth_seqs.append(pooled_targets[:,:,1].tolist())
            forecast_seqs.append(softmax_outputs[:,:,1].tolist())
        
        # save output
        if save_output:
            argmax_outputs, softmax_outputs =\
                sess.run([model.argmax_outputs, model.softmax_outputs], feed_dict={
                model.inputs: batch_x_test,
                model.features: batch_feature,
            })
            utils.crop_and_save(batch_x_test, softmax_outputs, pooling_size, 
                                   image_normalize=255.0,is_save=True, output_dir=output_dir,
                                    batch_number=i, is_thrmax=True, thr=0.1)
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
        if test_over_under:
            argmax_targets, softmax_outputs, acc = sess.run([model.argmax_targets, model.softmax_outputs, model.accuracy], 
                    feed_dict={
                        model.inputs: batch_x_test,
                        model.targets: batch_y_test,
                        model.features: batch_feature,
                        })
            softmaxs.append(softmax_outputs)
            targets.append(batch_y_test)
            acc_value += acc
            
    if test_roc:
        # ROC
        total_roc = utils.ROC(truth_seqs, forecast_seqs, neighborhood=1)

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
        acc_value /= test_loader.batch_number
        print acc_value
        utils.over_and_under_rate(softmaxs, targets, pooling_size=1)
