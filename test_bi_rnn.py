import time
import os
import sys

import tensorflow as tf
import numpy as np
import utils
from model import BiRnnModel

arg = sys.argv[1]
epoch = int(sys.argv[2])

# Constants
model_dir = arg
# epoch = 0
model_file = "model.ckpt-%d" % (epoch)
testID = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + \
         "_" + str(os.getpid())
logFilename = os.path.join(model_dir, "test_%d_%s.txt" % (epoch, testID))
batch_size = 32
# save ctc flag
save_decode = True
image_count = 1
ctc_dir = os.path.join(model_dir, 'ctc')
if not os.path.exists(ctc_dir):
    os.mkdir(ctc_dir)

# Loading the data
test_loader = utils.Loader('../psl_data/father/testdata_64',['data_0','data_1','data_2'], batch_size)

def LOG(Str):
    f1 = open(logFilename, "a")
    print Str
    print >> f1, Str
    f1.close()

LOG("Test ID: " + str(testID))

# THE MAIN CODE!
with tf.device('/gpu:1'):

    model = BiRnnModel(batch_size)

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
        tmp_err, tmp_ctc_err, tmp_logits_prob = sess.run([model.err, model.cost, model.logits_prob], feed_dict=feed)
        if save_decode:
            for logits_prob in tmp_logits_prob:
                prob_file = os.path.join(ctc_dir, 'ctc_prob_ep%d_%d.txt' % (epoch, image_count))
                image_count += 1
                with open(prob_file, 'w') as f:
                    for item in logits_prob:
                        f.write(str(item[-1]))
                        f.write('\n')

        test_ctc_err += tmp_ctc_err * np.shape(batch_x_test)[0]
        test_err += tmp_err
        LOG("Batch: {:3d}/{:3d}, Batch Label Error: {}/{}, Batch CTC Error: {:.5f}, Batch Time = {:.3f}"
            .format(i+1, test_loader.batch_number, tmp_err, batch_tar_len, tmp_ctc_err, time.time() - batch_start))
    test_ctc_err /= test_loader.train_length
    test_err /= test_loader.target_len
    # Calculate accuracy on the whole testing set
    LOG("Test Label Error: {:.5f}, Test CTC Error: {:.5f}, Time = {:.3f}"
        .format(test_err, test_ctc_err, time.time() - start))
