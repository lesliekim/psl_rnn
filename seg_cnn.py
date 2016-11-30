import seg_utils as utils
import tensorflow as tf
from model import SegNet_crop as Net
import numpy as np
import time
import os

# Training configs
num_epochs = 10
batch_size = 16 # image width are not same, so batch size can only be "1"
save_step = 100
trainID = "{}_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), str(os.getpid()))
save_dir = "./cnn_model/{}".format(trainID)

# Make save direction, save model in folder cnn_model
if not os.path.exists("./cnn_model"):
    os.mkdir("./cnn_model")
os.mkdir(save_dir)

# Loading the data
train_loader = utils.Loader('../psl_data/seg_cnn/traindata',['data_0', 'data_1', 'data_2', 'data_3'],batch_size)
test_loader = utils.Loader('../psl_data/seg_cnn/testdata_times',['data_0','data_1','data_2','data_3'],batch_size)

# train
with tf.Graph().as_default():
    sess = tf.Session()
    model = Net(batch_size)
    saver = tf.train.Saver()
    tf.get_variable_scope().reuse_variables()
    evaluation = Net(batch_size, is_test=True)
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(num_epochs):
        print("Processing epoch " + str(i))
        for batch in xrange(train_loader.batch_number):
            batch_x_train, batch_y_train, batch_steps_train, _ = train_loader.next_batch()
            seg_batch_x_train, seg_batch_y_train = utils.crop_image(batch_x_train,
                    batch_y_train, 32)
            if np.array(batch_x_train).shape[2] != np.array(batch_y_train).shape[1]:
                print np.array(batch_x_train).shape
                print np.array(batch_y_train).shape
                print "batch: " + str(batch)
            feed = {
                    model.inputs: seg_batch_x_train,
                    model.targets: seg_batch_y_train,
                    }
            _, loss_value, acc_val = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed)
            #print "outputs: {}".format(soft_outputs)
            #print "targets: {}".format(targets)
            if batch % 1000 == 0:
                print("Epoch %d, Batch %d: loss = %0.4f  acc = %0.4f" % (i, batch, loss_value, acc_val))

        print("Epoch %d: loss = %0.4f" % (i, loss_value))
        save_name = saver.save(sess, os.path.join(save_dir, "model"), global_step=i+1)

        # todo: add evaluation result saver
        acc_value = 0.0
        for batch in xrange(test_loader.batch_number):
            batch_x_test, batch_y_test, batch_steps_test, _ = test_loader.next_batch()
            #seg_batch_x_test, seg_batch_y_test = utils.crop_image(batch_x_test,
            #            batch_y_test, 32)

            acc, argmax_outputs, argmax_targets= sess.run([evaluation.accuracy, evaluation.argmax_outputs, evaluation.argmax_targets], feed_dict={
                evaluation.inputs: batch_x_test,
                evaluation.targets: batch_y_test,
                })
            acc_value += acc
        print "argmax outputs: {}".format(np.reshape(argmax_outputs, [batch_size, -1])[0, :])
        print "argmax targets: {}".format(np.reshape(argmax_targets, [batch_size, -1])[0, :])
        acc_value /= test_loader.batch_number
        print("test acc = %0.4f" % acc_value)

