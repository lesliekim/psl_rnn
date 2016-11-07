import os
import cv2 as cv
import utils

epoch = int(sys.arg[0])

pooling_size = 1# if RNN, set this parameter to 1
data_dir = '/home/jia/psl/tf_rnn/psl_data/gulliver/testdata'
ctc_dir = '/home/jia/psl/tf_rnn/psl_rnn/model/gulliverRNNonly/ctc'

assert os.path.isdir(data_dir)
assert os.path.isdir(ctc_dir)

# make sure your image data order matchs your ctc file order
test_data = ['data_0','data_1','data_2','data_3']
test_loader = utils.Loader(data_dir, test_data, 512)

image_count = 1 # make sure its initial number is the same with test_bi_rnn.py!!!
for i in xrange(test_loader.batch_number):
    print('processing batch %d' % (i + 1))

    batch_img = test_loader.next_batch()
    for img in batch_img:
        if image_count % 100 == 0:
            print('finish %d images' % (image_count))
        ctc_file_name = 'ctc_prob_ep%d_%d.txt' % (epoch, image_count)
        new_image_name = 'img_ep%d_%d.jpg' % (epoch, image_count)
        image_count += 1
        ctc_prob = utils.read_probfile(os.path.join(ctc_dir, ctc_file_name))
        # calulate position from probability
        ctc_pos = utils.prob_to_pos(ctc_prob, pooling_size)
        # draw 
        draw_pos_on_image(ctc_pos, img, os.path.join(ctc_dir, new_image_name))

