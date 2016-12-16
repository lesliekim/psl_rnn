import time
import os
import sys

import tensorflow as tf
import numpy as np
import utils
import seg_utils
from model import BiRnnModel

arg = sys.argv[1]
epoch = int(sys.argv[2])

# Constants
model_dir = arg
# epoch = 0
model_file = "model.ckpt-%d" % (epoch)
output_dir = '../psl_data/father'
# batch size
batch_size = 1

# Loading the data
test_loader = utils.Loader('../psl_data/father/seg_synthesis_traindata',\
        ['data_4'], batch_size)

# THE MAIN CODE!
with tf.device('/gpu:1'):

    model = BiRnnModel(batch_size)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2

def LOG(Str):
    print Str

LOG("Launching the graph...")
match_seq = []
match_line_number = 0
truth_line_number = 0
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
        if i % 100 == 0:
            print "processing {}".format(i)

        batch_start = time.time()
        batch_x_test, batch_y_test, batch_steps_test, batch_tar_len \
            = test_loader.next_batch()
        feed = {model.inputs: batch_x_test,
                #model.targets: batch_y_test,
                model.seq_len: batch_steps_test}

        tmp_logits_prob = sess.run([model.logits_prob], feed_dict=feed)
        ctc_prob = [1.0 - x[-1] for x in tmp_logits_prob[0][0]]
        label = batch_y_test[1].tolist()
        # match
        batch_match_seq, batch_match_line_number = \
                seg_utils.match(label, ctc_prob, 4)
        
        match_seq += batch_match_seq
        match_line_number += batch_match_line_number
        truth_line_number += label.count(1)

roc_line = seg_utils.single_roc_line(match_seq, match_line_number, truth_line_number)
# save file
with open(os.path.join(output_dir, 'roc_1.csv'),'w') as f:
    for item in roc_line:
        f.write(str(item[0]))
        f.write(',')
        f.write(str(item[1]))
        f.write(',')
        f.write(str(item[2]))
        f.write('\n')
