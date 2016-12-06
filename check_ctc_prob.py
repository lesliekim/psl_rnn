import os
import sys
import cv2 as cv
import utils

epoch = int(sys.argv[1])

padding = 0.189871 # you should change its value every time
pooling_size = 1 # if RNN, set this parameter to 1
data_dir = '/home/jia/psl/tf_rnn/psl_data/father/testdata_64'
ctc_dir = '/home/jia/psl/tf_rnn/psl_rnn/model/gulliverRNNonly/ctc'
output_dir = os.path.join(ctc_dir, 'image')

assert os.path.isdir(data_dir)
assert os.path.isdir(ctc_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# make sure your image data order matchs your ctc file order
test_data = ['data_0','data_1','data_2']
test_loader = utils.Loader(data_dir, test_data, 1)

image_count = 1 # make sure its initial number is the same with test_bi_rnn.py!!!
batch_img = test_loader.image
for img in batch_img:
    if image_count % 100 == 0:
        print('finish {}images'.format(image_count))
    ctc_file_name = 'ctc_prob_ep{}_{}.txt'.format(epoch, image_count)
    new_image_name = 'img_ep{}_{}.jpg'.format(epoch, image_count)
    image_base = 'img_ep{}_{}_'.format(epoch, image_count)
    image_count += 1
    ctc_prob = utils.read_probfile(os.path.join(ctc_dir, ctc_file_name))
    # calulate position from probability
    ctc_pos = utils.prob_to_pos(ctc_prob, padding=padding, pooling_size=pooling_size)
    '''
    print "ctc prob len " + str(len(ctc_prob))
    print "ctc pos len " + str(len(ctc_pos))
    print "image shape " + str(img.shape)
    '''
    # draw 
    #utils.draw_pos_on_image(ctc_pos, img, os.path.join(output_dir, new_image_name))
    
    # segment result for later process
    utils.seg_image(ctc_pos, img, image_base, output_dir)
