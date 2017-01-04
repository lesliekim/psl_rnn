import classify_utils as utils
import tensorflow as tf
from model import ClassifyCnnNet as Net
import numpy as np
import time
import os

# Training configs
num_epochs = 20
batch_size = 32 # image width are not same, so batch size can only be "1"
save_step = 100
trainID = "{}_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), str(os.getpid()))
save_dir = "./classify_cnn_model/{}".format(trainID)

# Make save direction, save model in folder cnn_model
if not os.path.exists("./classify_cnn_model"):
    os.mkdir("./classify_cnn_model")
os.mkdir(save_dir)

# copy param file
os.system("cp ./classify_cnn.py " + save_dir)
os.system("cp ./classify_param.py " + save_dir)
os.system("cp ./model.py " + save_dir)

# log train and test info
logFilename = "{}/log.txt".format(save_dir)
def LOG(Str):
    f = open(logFilename, 'a')
    print Str
    print >> f, Str
    f.close()


# Loading the data
train_loader = utils.Loader('../psl_data/classify_cnn/test_single_traindata',['data_0'],batch_size)
test_loader = utils.Loader('../psl_data/classify_cnn/test_single_traindata',['data_0'],batch_size)

# train
with tf.Graph().as_default():
    sess = tf.Session()
    model = Net(batch_size)
    saver = tf.train.Saver(max_to_keep=0)
    tf.get_variable_scope().reuse_variables()
    evaluation = Net(batch_size, is_test=True)
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(num_epochs):
        LOG("Processing epoch " + str(i))
        for batch in xrange(train_loader.batch_number):
            batch_x_train, batch_y_train = train_loader.next_batch()
            feed = {
                    model.inputs: batch_x_train,
                    model.targets: batch_y_train,
                    }
            _, loss_value, acc_val = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed)

            if batch % 1000 == 0:
                LOG("Epoch %d, Batch %d: loss = %0.4f  acc = %0.4f" % (i, batch, loss_value, acc_val))

        save_name = saver.save(sess, os.path.join(save_dir, "model"), global_step=i+1)

        # todo: add evaluation result saver
        acc_value = 0.0
        for batch in xrange(test_loader.batch_number):
            batch_x_test, batch_y_test = test_loader.next_batch()

            acc, argmax_outputs, argmax_targets = sess.run([evaluation.accuracy, evaluation.argmax_outputs, evaluation.argmax_targets], feed_dict={
                evaluation.inputs: batch_x_test,
                evaluation.targets: batch_y_test,
                })
            acc_value += acc

        acc_value /= test_loader.batch_number
        LOG("test acc = %0.4f" % acc_value)

