import seg_utils as utils
import tensorflow as tf
from model import RecNet as Net
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
train_loader = utils.Loader('../psl_data/seg_cnn/traindata_total',
                                                ['data_0', 'data_1', 'data_2', 'data_3', 
                                                'data_5', 'data_6', 'data_7', 'data_9', 'data_10', 'data_11', 
                                                'data_13', 'data_14', 'data_15'],batch_size)
test_loader = utils.Loader('../psl_data/seg_cnn/traindata_total',['data_4','data_8','data_12'],batch_size)

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
            #seg_batch_x_train, seg_batch_y_train = utils.crop_image(batch_x_train,
            #        batch_y_train, 32)
            feed = {
                    model.inputs: batch_x_train,
                    model.targets: batch_y_train,
                    }
            _, loss_value, acc_val = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed)

            if batch % 1000 == 0:
                print("Epoch %d, Batch %d: loss = %0.4f  acc = %0.4f" % (i, batch, loss_value, acc_val))

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
        for kk in xrange(5):
            print "argmax outputs: {}".format(np.reshape(argmax_outputs, [batch_size, -1])[kk, :])
            print "argmax targets: {}".format(np.reshape(argmax_targets, [batch_size, -1])[kk, :])
        acc_value /= test_loader.batch_number
        print("test acc = %0.4f" % acc_value)

