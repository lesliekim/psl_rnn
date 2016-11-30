import seg_utils as utils
import tensorflow as tf
from model import SegNet_crop as Net
import os
import sys

# Training configs
model_dir = sys.argv[1]
epoch = int(sys.argv[2])

pooling_size = 4

model_file = os.path.join(model_dir, 'model-{}'.format(epoch))
batch_size = 16 # if image width are not same, batch size can only be "1"
output_dir = '../psl_data/seg_cnn/crop_image'
# Loading the data
test_loader = utils.Loader('../psl_data/seg_cnn/traindata',['data_2'],batch_size)

# test
with tf.Graph().as_default():
    sess = tf.Session()
    model = Net(batch_size, is_test=True)
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    sess.run(init)

    print "Loading pb file from: {}".format(model_file)
    saver.restore(sess, model_file)

    for i in xrange(test_loader.batch_number):
        #print "Processing batch {} / {}".format(i, test_loader.batch_number)
        # todo: add evaluation result saver
        batch_x_test, batch_y_test, batch_steps_test, _ = test_loader.next_batch()

        argmax_outputs = sess.run([model.argmax_outputs], feed_dict={
            model.inputs: batch_x_test,
            #model.targets: batch_y_test,
        })

        # save output
        utils.crop_and_save(batch_x_test, argmax_outputs[0], output_dir, i, pooling_size)
        
