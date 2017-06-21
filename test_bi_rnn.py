import time
import os
import sys

import tensorflow as tf
import numpy as np
import  utils
from model import CnnRnnModel

arg = sys.argv[1]
epoch = int(sys.argv[2])

# Constants
model_dir = arg
# epoch = 0
model_file = "model.ckpt-%d" % (epoch)
testID = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + "_" + str(os.getpid())
logFilename = os.path.join(model_dir, "test_%d_%s.txt" % (epoch, testID))

# batch size
batch_size = 32

# save  flag
save_decode = False
image_count = 1
output_count = 1
prob_folder_name = 'ctc' # it may not always be ctc
prob_dir = os.path.join(model_dir, prob_folder_name)
if save_decode and (not os.path.exists(prob_dir)):
    os.mkdir(prob_dir)

# Loading the data

test_loader = utils.Loader('../psl_data/gulliver/testdata',\
        ['data_0','data_1','data_2','data_3'], batch_size)
'''
test_loader = utils.Loader('../psl_data/English_card_hard/email_binary_sub_traindata',\
        ['data_0','data_1'], batch_size)
'''
def LOG(Str):
    f1 = open(logFilename, "a")
    print Str
    print >> f1, Str
    f1.close()

LOG("Test ID: " + str(testID))

# THE MAIN CODE!
with tf.device('/gpu:1'):

    model = CnnRnnModel(batch_size)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2

LOG("Launching the graph...")
with tf.Session(config=config) as sess:
    # Initializate the weights and biases
    tf.initialize_all_variables().run()
    if epoch <= 0:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            LOG("Loading pb file from: " + ckpt.model_checkpoint_path)
        else:
            LOG('no checkpoint found...')
    else:
        LOG("Loading pb file from: %s" % (os.path.join(model_dir,model_file)))
        model.saver.restore(sess, os.path.join(model_dir, model_file))
    test_err = 0
    test_ctc_err = 0
    start = time.time()
    for i in xrange(test_loader.batch_number):
        batch_start = time.time()
        batch_x_test, batch_y_test, batch_steps_test, batch_tar_len \
            = test_loader.next_batch()
        feed = {model.inputs: batch_x_test,
                model.targets: batch_y_test,
                model.seq_len: batch_steps_test}

        if not save_decode:
            tmp_err, tmp_ctc_err = sess.run([model.err, model.cost], feed_dict=feed)
        else:
            tmp_err, tmp_ctc_err, tmp_logits_prob = sess.run([model.err, model.cost, model.logits_prob], feed_dict=feed)
        
            for logits_prob in tmp_logits_prob:
                prob_file = os.path.join(prob_dir, '%s_prob_ep%d_%d.txt' % (prob_folder_name, epoch, image_count))
                image_count += 1
                with open(prob_file, 'w') as f:
                    for item in logits_prob:
                        f.write(str(item[-1])) # for ctc probablity
                        #f.write(str(item))
                        f.write('\n')
            '''
            tmp_err, tmp_ctc_err, tmp_decoded = sess.run([model.err, model.cost, model.decoded[0]], feed_dict=feed)
            
            for j in xrange(0, batch_size):
                tar = utils.get_row(batch_y_test, j)
                truth_filename = os.path.join(prob_dir, 'ep{}_{}.std'.format(epoch, image_count))
                image_count += 1
                with open(truth_filename, 'w') as f:
                    for item in tar:
                        f.write(chr(item))

            for j in xrange(0, batch_size):
                decoded_str = utils.get_row(tmp_decoded, j)
                result_filename = os.path.join(prob_dir, 'ep{}_{}.txt'.format(epoch, output_count))
                output_count += 1
                with open(result_filename, 'w') as f:
                    for item in decoded_str:
                        f.write(chr(item))
            '''

        test_ctc_err += tmp_ctc_err * np.shape(batch_x_test)[0]
        test_err += tmp_err

        LOG("Batch: {:3d}/{:3d}, Batch Label Error: {}/{}, Batch CTC Error: {:.5f}, Batch Time = {:.3f}"
            .format(i+1, test_loader.batch_number, tmp_err, batch_tar_len, tmp_ctc_err, time.time() - batch_start))

    test_ctc_err /= test_loader.train_length
    test_err /= test_loader.target_len
    #test_err /= test_loader.batch_number
    # Calculate accuracy on the whole testing set
    LOG("Test Label Error: {:.5f}, Test CTC Error: {:.5f}, Time = {:.3f}"
        .format(test_err, test_ctc_err, time.time() - start))
