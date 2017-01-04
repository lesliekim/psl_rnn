import time
import os

import utils
from model import BiRnnModel
import tensorflow as tf
import numpy as np


# Constants
trainID = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + \
         "_" + str(os.getpid())
checkpoint_dir = "./model/" + trainID
os.mkdir(checkpoint_dir)
os.system("cp ./model.py " + checkpoint_dir)
os.system("cp ./bi_rnn.py " + checkpoint_dir)
os.system("cp ./test_bi_rnn.py" + checkpoint_dir)
os.system("cp ./param.py" + checkpoint_dir)
logFilename = checkpoint_dir + "/log_" + trainID + ".txt"
model_name = "English" + "_" + trainID


# Training configs
num_epochs = 20000
batch_size = 16
disp_steps = 1000
checkpoint_steps = 1

keep_prob_1 = 1.0
keep_prob_2 = 1.0
keep_prob_3 = 1.0

# continue training
continue_train = 1 
model_dir = './model/244ImagesRNNonly_new'

# Loading the data

train_loader = utils.Loader('../psl_data/244Images/traindata_sub',['data_0','data_1','data_2','data_3','data_4','data_5','data_6','data_7','data_8','data_9','data_10','data_11','data_12'], batch_size)

def LOG(Str):
    f = open(logFilename, "a")
    print Str
    print >> f, Str
    f.close()

LOG('trainID = '+str(trainID))
LOG('batch_Size = '+str(batch_size))
LOG('target len = ' + str(train_loader.target_len))
LOG('keep prob:')
LOG(keep_prob_1)
LOG(keep_prob_2)
LOG(keep_prob_3)
# THE MAIN CODE!

with tf.device('/cpu:0'):
    model = BiRnnModel(batch_size)


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

LOG("Launching the graph...")

with tf.device('/gpu:0'):
    with tf.Session(config=config) as session:
    #with tf.Session() as session:
        # train_writer = tf.train.SummaryWriter("./model/graphDef/" + model_name,
        #                                       session.graph)
        # Initializate the weights and biases
        tf.initialize_all_variables().run()
        tf.train.write_graph(session.graph_def, checkpoint_dir,
                             'graph_def.pb', as_text=False)
        if continue_train:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                LOG('load model from: '+ ckpt.model_checkpoint_path)
                model.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                LOG('no checkpoint found...')

        for curr_epoch in xrange(num_epochs):
            train_cost = train_err = 0
            start = time.time()

            for batch in xrange(train_loader.batch_number):
                batch_timer = time.time()
                # get batch data
                batch_x_train, batch_y_train, batch_steps_train, batch_tar_len\
                    = train_loader.next_batch()

                feed = {model.inputs: batch_x_train,
                        model.targets: batch_y_train,
                        model.seq_len: batch_steps_train}

                # model.err op has been removed to speed up training !
                batch_cost, _ =\
                    session.run([model.cost, model.optimizer], feed)

                #print "argmax logits: {}".format(argmax_logits.shape)
                #print "argmax targets: {}".format(argmax_targets.shape)

                train_cost += batch_cost*np.shape(batch_x_train)[0]
                #train_err += batch_err

                if batch % disp_steps == 0:
                    #LOG('Batch {}/{}, batch_cost = {:.3f}, batch_err = {}/{}, Time: {:.3f}'
                    #    .format(batch, train_loader.batch_number, batch_cost, int(batch_err), batch_tar_len, time.time() - batch_timer))
                    LOG('Batch {}/{}, batch_cost = {:.3f}, Time: {:.3f}'
                        .format(batch, train_loader.batch_number, batch_cost, time.time() - batch_timer))
                    '''
                    # Decoding
                    decoded_Array = session.run(model.decoded[0], feed_dict=feed)
                    decode_len = 3
                    LOG('Target:')
                    for i in range(0, min(batch_size, decode_len)):
                        tar = utils.get_row(batch_y_train, i)
                        LOG(tar)
                    LOG('Decoded:')
                    for i in range(0, min(batch_size, decode_len)):
                        decoded_str = utils.get_row(decoded_Array, i)
                        LOG(decoded_str)
                    '''

            train_cost /= train_loader.train_length
            train_err /= train_loader.target_len

            log = trainID + "  Epoch {}/{}, train_cost = {:.3f}, time = {:.3f}"
            LOG(log.format(curr_epoch+1, num_epochs, train_cost, time.time() - start))

            if (curr_epoch + 1) % checkpoint_steps == 0:
                model.saver.save(session, checkpoint_dir + '/model.ckpt', global_step=curr_epoch+1)
                model.saver.as_saver_def()
                os.system('python test_bi_rnn.py ' + checkpoint_dir + ' ' + str(curr_epoch+1))
            train_loader.shuffle()
