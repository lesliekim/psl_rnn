import os
import editdistance
import cv2 as cv
import numpy as np
import time
import seg_utils as utils
import classify_utils as cutils
import seg_param as param
import tensorflow as tf
from model import SegCnnNet, SegCnnNet_space, ClassifyCnnNet

# Test configs
segCnnNet_space_model_dir = "./cnn_model/segCnnNet_1_pad2_train2"
segCnnNet_space_model_epoch = 10
segCnnNet_model_dir = "./cnn_model/segCnnNet_total_random"
segCnnNet_model_epoch = 10
classifyCnnNet_model_dir = "./classify_cnn_model/base_net"
classifyCnnNet_model_epoch = 10

# Loading the data
batch_size = 1
test_loader = utils.Loader('../psl_data/father/traindata_32', ['data_4'], batch_size)

g = tf.Graph()
sess = tf.InteractiveSession(graph=g)

# Loading Graph and Test
lines_1 = []
lines_2 = []
lines_3 = []
ground_truth = []
outdir_1 = '../psl_data/paraline/step1'
outdir_2 = '../psl_data/paraline/step2'
outdir_3 = '../psl_data/paraline/step3'
if not os.path.exists(outdir_1):
    os.mkdir(outdir_1)
if not os.path.exists(outdir_2):
    os.mkdir(outdir_2)
if not os.path.exists(outdir_3):
    os.mkdir(outdir_3)

#with tf.Graph().as_default() as g:
with g.as_default():
    model = SegCnnNet_space(batch_size, is_test=True)
    saver = tf.train.Saver()

    print "Loading model SegCnnNet_space......"
    saver.restore(sess, os.path.join(segCnnNet_space_model_dir, \
            'model-{}'.format(segCnnNet_space_model_epoch)))

    for i in xrange(test_loader.batch_number):
        batch_x_test, batch_y_test,_,_ = test_loader.next_batch()
        argmax_outputs = sess.run([model.argmax_outputs], feed_dict={
            model.inputs: batch_x_test,
            })
        lines_1.append(utils.word_segmentation(batch_x_test, argmax_outputs[0], 1, 1.0,is_save=False))
        '''
        # save for check
        for ii,img in enumerate(lines_1[i]):
            img_name = "line_{}_word_{}.bin.png".format(i*batch_size, ii)
            cv.imwrite(os.path.join(outdir_1, img_name), img*255.)
        '''
        # move batch_y_test padding
        no_pad_y = []
        for y in batch_y_test:
            no_pad_y.append(y[:np.where(y==param.pad_value)[0][0]].tolist())

        ground_truth += no_pad_y

    #tf.reset_default_graph()

g = tf.Graph()
sess = tf.InteractiveSession(graph=g)

with g.as_default():
    model = SegCnnNet(is_test=True)
    saver = tf.train.Saver()

    print "Loading model SegCnnNet......"
    saver.restore(sess, os.path.join(segCnnNet_model_dir, \
            'model-{}'.format(segCnnNet_model_epoch)))

    for j,line in enumerate(lines_1):
        one_line = []
        for jj, word in enumerate(line):
            batch_x_test = np.reshape(word, [1, word.shape[0], word.shape[1], 1])
            argmax_outputs = sess.run([model.argmax_outputs], feed_dict={
                model.inputs: batch_x_test,
                })
            one_line.append(utils.crop_and_save(batch_x_test, argmax_outputs[0], 4, 1.0,is_save=False))
            '''
            # save image for check
            for jjj, img in enumerate(one_line[jj][0]):
                img_name = "line_{}_word_{}_char_{}.bin.png".format(j, jj, jjj)
                cv.imwrite(os.path.join(outdir_2, img_name), img*255.)
            ''' 
        lines_2.append(one_line)

g = tf.Graph()
sess = tf.InteractiveSession(graph=g)

with g.as_default():
    model = ClassifyCnnNet(is_test=True)
    saver = tf.train.Saver()

    print "Loading model ClassifyCnnNet......"
    saver.restore(sess, os.path.join(classifyCnnNet_model_dir, \
            'model-{}'.format(classifyCnnNet_model_epoch)))

    for k,line in enumerate(lines_2):
        line_out = []
        for kk,word in enumerate(line):
            batch_x_test = []
            for kkk,char in enumerate(word[0]):
                '''
                img_name = "line_{}_word_{}_char_{}.bin.png".format(k, kk, kkk)
                cv.imwrite(os.path.join(outdir_3, img_name), cutils.pad_to_square(np.asarray(char))*255.)
                '''
                batch_x_test.append(cutils.pad_to_square(np.asarray(char)))

            batch_x_test = np.asarray(batch_x_test)
            s = batch_x_test.shape
            batch_x_test = np.reshape(batch_x_test, [s[0],s[1],s[2],1])
            argmax_outputs = sess.run([model.argmax_outputs], feed_dict={
                model.inputs: batch_x_test,
                })
            '''
            for char, a in zip(word[0], argmax_outputs[0].tolist()):
                img_name = "line_{}_word_{}_char_{}.bin.png".format(k, kk, a)
                cv.imwrite(os.path.join(outdir_3, img_name), cutils.pad_to_square(np.asarray(char))*255.)
            '''
            line_out += argmax_outputs[0].tolist()
            if kk != len(line):
                line_out.append(ord(' '))

        lines_3.append(line_out)
        out_string = ''.join(map(chr, line_out))
        truth_string = ''.join(map(chr, ground_truth[k]))
        print "out_string: ", out_string
        print "truth_string: ", truth_string
        print editdistance.eval(out_string, truth_string)

        


