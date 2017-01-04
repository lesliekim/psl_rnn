from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from model import TestNet
import numpy as np
import time
import os

with tf.Graph().as_default():
    num_steps = 2000
    save_step = 100
    trainID = "{}_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), str(os.getpid()))
    save_dir = "./cnn_model/{}".format(trainID)
    if not os.path.exists("./cnn_model"):
        os.mkdir("./cnn_model")
    os.mkdir(save_dir)

    sess = tf.Session()
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False)

    model = TestNet()
    saver = tf.train.Saver()
    tf.get_variable_scope().reuse_variables()
    evaluation = TestNet(is_test=True)
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(num_steps):
        batch = mnist.train.next_batch(100)
        feed = {
                model.inputs: batch[0],
                model.targets: batch[1]
                }
        _, loss_value, acc_value = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed)
        if i % 100 == 0:
            print("Step %d: loss = %0.4f acc = %0.4f" % (i, loss_value, acc_value))
            
        if (i+1) % save_step == 0 or i+1 == num_steps:
            save_name = saver.save(sess, os.path.join(save_dir, "model"), global_step=i+1)

    acc_value = sess.run([evaluation.accuracy], feed_dict={
        evaluation.inputs: mnist.test.images,
        evaluation.targets: mnist.test.labels
        })
    print("test acc = %0.4f" % acc_value[0])

