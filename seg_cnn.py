import seg_utils as utils
import tensorflow as tf
from model import SegCnnNet as Net
import numpy as np
import time
import os
import seg_param as param

np.set_printoptions(threshold='nan')
# Training configs
image_height = param.height
num_epochs = 150
batch_size = 32 # image width are not same, so batch size can only be "1"
save_step = 100
trainID = "{}_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()), str(os.getpid()))
save_dir = "./cnn_model/{}".format(trainID)

# continue training
continue_train = False
model_dir = './cnn_model/gulliver4_base'

# Make save direction, save model in folder cnn_model
if not os.path.exists("./cnn_model"):
    os.mkdir("./cnn_model")
os.mkdir(save_dir)

# copy param file
os.system("cp ./seg_cnn.py " + save_dir)
os.system("cp ./seg_param.py " + save_dir)
os.system("cp ./model.py " + save_dir)

# log train and test info
logFilename = "{}/log.txt".format(save_dir)
def LOG(Str):
    f = open(logFilename, 'a')
    print Str
    print >> f, Str
    f.close()

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

# Loading the data
train_loader = utils.Loader('../psl_data/seg_cnn/traindata_gulliver_3',
                        ['data_{}'.format(x) for x in xrange(17)],batch_size)
test_loader = utils.Loader('../psl_data/seg_cnn/traindata_father_4',['data_0'],batch_size)

# train
with tf.Graph().as_default():
    sess = tf.Session()
    model = Net()
    saver = tf.train.Saver(max_to_keep=0)
    tf.get_variable_scope().reuse_variables()
    evaluation = Net(is_test=True)
    init = tf.initialize_all_variables()
    sess.run(init)

    if continue_train:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            LOG('load model from: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            LOG('no checkpoint fount...')

    for i in range(num_epochs):
        LOG("Processing epoch " + str(i))
        for batch in xrange(train_loader.batch_number):
            batch_x_train, batch_y_train, batch_steps_train, _ = train_loader.next_batch() #seg_batch_x_train, seg_batch_y_train = utils.crop_image(batch_x_train,batch_y_train, 32)
            
            # make base features
            batch_feature = np.reshape(np.sum(batch_x_train, axis=1), [batch_x_train.shape[0], 1, -1, 1])
            # noramlize base features to [0,1]
            batch_feature = norm(batch_feature)
            
            feed = {
                    model.inputs: batch_x_train,
                    model.targets: batch_y_train,
                    model.features: batch_feature,
                    }
            _, loss_value, acc_val = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed)
            if batch % 1000 == 0:
                LOG("Epoch %d, Batch %d: loss = %0.4f  acc = %0.4f" % (i, batch, loss_value, acc_val))

        save_name = saver.save(sess, os.path.join(save_dir, "model"), global_step=i+1)

        # todo: add evaluation result saver
        acc_value = 0.0
        for batch in xrange(test_loader.batch_number):
            batch_x_test, batch_y_test, batch_steps_test, _ = test_loader.next_batch()#seg_batch_x_test, seg_batch_y_test = utils.crop_image(batch_x_test,batch_y_test, 32)
            
            # make base features
            batch_feature_test = np.reshape(np.sum(batch_x_test, axis=1), [batch_x_test.shape[0], 1, -1, 1])
            # noramlize base features to [0,1]
            batch_feature_test = norm(batch_feature_test)
            
            acc, argmax_outputs, argmax_targets, softmax_y = sess.run([evaluation.accuracy, evaluation.argmax_outputs, evaluation.argmax_targets, evaluation.softmax_outputs], feed_dict={
                evaluation.inputs: batch_x_test,
                evaluation.targets: batch_y_test,
                evaluation.features: batch_feature_test,
                })
            acc_value += acc
        for kk in xrange(5):
            LOG("argmax outputs: {}".format(np.reshape(argmax_outputs, [batch_size, -1])[kk, :]))
            LOG("argmax targets: {}".format(np.reshape(argmax_targets, [batch_size, -1])[kk, :]))
        acc_value /= test_loader.batch_number
        LOG("test acc = %0.4f" % acc_value)

