import os
import pickle
import random
import math
import numpy as np
import seg_param as param
import cv2 as cv

norm_height = param.height

'''
Data Loader
'''
def load_set(datadir, dataname):
    filename = os.path.join(datadir, dataname + '_image.p')
    print('Loading data from: ' + filename)
    f = open(filename, 'rb')
    seqImage = pickle.load(f)
    f.close()

    filename = os.path.join(datadir, dataname + '_label.p')
    print('Loading label from: ' + filename)
    f = open(filename, 'rb')
    seqLabel = pickle.load(f)
    f.close()
    return seqImage, seqLabel
def sparse_array(sequences, width, on_value, off_value, dtype=np.int32):
    indices = []

    for i,seq in enumerate(sequences):
        ans = [[on_value, off_value] for k in xrange(width)]
        for j,item in enumerate(seq):
            if item > 0:
                ans[j][1] = on_value
                ans[j][0] = off_value
        indices.append(ans)

    indices = np.asarray(indices, dtype=np.int64)
    return indices

def pad_array(sequences, width):
    pad_value = param.pad_value
    indices = []
    for i,seq in enumerate(sequences):
        ans = [pad_value] * width
        ans[:len(seq)] = seq[:]
        indices.append(ans)

    indices = np.asarray(indices, dtype=np.int64)
    return indices

def get_row(sparse_tuple, row, dtype=np.int32):
    optlist = []
    cnt = 0
    for pos in sparse_tuple[0]:
        if pos[0] == row:
            optlist.append(sparse_tuple[1][cnt])
        cnt += 1
    return optlist

class Loader(object):
    def __init__(self, datadir, set_list = [], batch_size = 1, is_sparse=False): 
        self.count = 0
        self.batch_size = batch_size
        self.norm_height = norm_height
        self.is_sparse = is_sparse
        print('Data Loader initializing ...')
        self.image = []
        self.label = []
        for dataset in set_list:
            print('Set ' + dataset + '...')
            tmpImage, tmpLabel = load_set(datadir, dataset)
            self.image += tmpImage
            self.label += tmpLabel
        self.train_length = len(self.label)
        self.batch_number = int(np.ceil(np.float(self.train_length) / np.float(batch_size)))
        
        self.target_len = 0
        for target in self.label:
            self.target_len += len(target)

    def next_batch(self):
        if self.count + self.batch_size <= self.train_length:
            x_batch_seq = self.image[self.count : self.count + self.batch_size]
            y_batch_seq = self.label[self.count : self.count + self.batch_size]
            self.count += self.batch_size
        else:
            left_num = self.count + self.batch_size - self.train_length
            x_batch_seq = self.image[self.count :] + self.image[0 : left_num]
            y_batch_seq = self.label[self.count :] + self.label[0 : left_num]
            self.count = 0

        step_batch = np.zeros(shape=[len(x_batch_seq)], dtype='int64')
        for i in xrange(len(step_batch)):
            step_batch[i] = np.shape(x_batch_seq[i])[1]

        max_width =int(np.ceil(np.max(step_batch)/8.0) * 8.0) # ensure image width is multiples of 8
        x_batch = np.zeros(shape=[len(step_batch), self.norm_height, max_width, 1])
        for i in xrange(len(step_batch)):
            x_batch[i,:, :step_batch[i], 0] = x_batch_seq[i]

        # Creating sparse representation to feed the placeholder
        if self.is_sparse:
            y_batch = sparse_array(y_batch_seq, max_width, on_value=1, off_value=0)
        else:
            y_batch = pad_array(y_batch_seq, max_width)

        tar_len_batch = 0
        for y in y_batch_seq:
            tar_len_batch += len(y)
        return x_batch / 255., y_batch, step_batch, tar_len_batch

    def shuffle(self):
        compact = zip(self.image, self.label)
        random.shuffle(compact)
        self.image, self.label = map(list, zip(*compact))

'''
Process CTC output
'''
def read_probfile(filename):
    with open(filename, 'r') as f:
        prob = [float(x[:-1]) for x in f]
    return prob

def move_padding(prob, padding=0.120172):# default value 0.189871 comes from really network output on padding
    N = len(prob)
    end = N - 1
    for i in xrange(N - 1, -1, -1):
        if prob[i] == padding:
            end = i
            break

    return prob[:end][:]

def local_min(prob, threshold=0, bg=1.0, fg=0.0):
    N = len(prob)
    if N < 2:
        raise ValueError('prob length should > 1!')

    peak = [bg] * N
    for i in xrange(N):
        if i > 0 and i < N - 1:
            if prob[i] < prob[i - 1] and prob[i] < prob[i + 1]:
                peak[i] = prob[i]
            elif i == 0:
                if prob[i] < prob[i + 1]:
                    peak[i] = prob[i]
            else:
                if prob[i] < prob[i - 1]:
                    peak[i] = prob[i]

    if threshold:
        for i in xrange(N):
            if peak[i] > threshold:
                peak[i] = bg # noise point, set to background value
            else:
                peak[i] = fg # local min

    return peak

def prob_to_pos(prob, pooling_size=1, bg=1.0, fg=0.0): # we assume your stride size is equal to your pooling size
    n_prob = move_padding(prob)
    peak = local_min(n_prob, 0.02)
    N = len(peak)
    pos = [bg] * (pooling_size * N)
    if pooling_size > 1:
        count = 0
        for i in xrange(N):
            for j in xrange(pooling_size):
                pos[count] = peak[i]
                count += 1
    else:
        pos = peak[:]

    return pos

def draw_pos_on_image(pos, img, img_name, fg=0.0):
    height, width = img.shape[:2]
    N = len(pos)
    if abs(N - width) > 2:
        raise ValueError('ctc prob length and image width are not match: prob length: %d, image width: %d' % (N, width))

    f_img = cv.cvtColor(np.cast['uint8'](img), cv.COLOR_GRAY2BGR)
    
    for i in xrange(N):
        if pos[i] == fg and i < width:
            cv.line(f_img, (i, 0), (i, height - 1), (0,0,255))
    
    cv.imwrite(img_name, f_img)
        
def peak_search(array, bg=0, fg=1):
    threshold = 0.5
    return [fg if x > threshold else bg  for x in array]

def reduce_length(array, times):
    ans = []
    N = len(array)
    begin = 0
    end = min(4, N)
    while begin < N:
        ans.append(max(array[begin:end]))
        begin = end
        end = min(end + 4, N)
    return ans

'''
Crop image for SegNet_Crop 
'''
def is_valid_segment(positions, center, threshold=10):
    for pos in positions:
        if abs(pos - center) < threshold:
            return False

    return True

def crop_image(image_seq, label_seq, crop_width, padding=0):
    '''
    crop image for network inputs
    image_seq: image sequence, list, each item is an image mat
    label_seq: label sequence, list, each item is an 1-D array which has 
                the same length with image's width (not one-hot format)
    crop_width: crop image width. Note that crop image's height is the same
                as its width
    padding: parameter for opencv, image padding
    return value: tuple, (crop_image, crop_label), crop_label is an one-hot array
    '''
    crop_image_seq = []
    crop_label_seq = []
    affine_dst = np.array([[0,0],[crop_width - 1,0],[crop_width - 1,crop_width - 1]], dtype=np.float32)
    for i,image in enumerate(image_seq):
        label_none_padding = move_padding(label_seq[i], padding=0)
        
        seg_positions = set([k for k,item in enumerate(label_none_padding) if item == 1])

        # crop positive samples
        for pos in seg_positions:
            affine_src = np.array([[pos + 1 - crop_width/2, 0],[pos + crop_width/2, 0],[pos + crop_width/2, crop_width - 1]],dtype=np.float32)
            affine_mat = cv.getAffineTransform(affine_src, affine_dst)
            crop_image = cv.warpAffine(image, affine_mat, dsize=(crop_width, crop_width))
            crop_image_seq.append(crop_image)
            crop_label_seq.append([0, 1])
        
        # crop negative samples
        count = 0
        # negative samples : positive samples
        radio = 1.0
        number_negative_samples = len(seg_positions) * radio + 1
        
        # Histogram based crop for negative samples
        # image : [height, width, channel], black background, white character
        hist = np.transpose(np.array(image).sum(axis=0).flatten()) # accumulate along heigth
        hist = move_padding(hist, padding=0.0)
        hist_min = local_min(hist, bg = -1) # local min of hist is greater than -1 while others equal to -1
        hist_min_pair = [(k, item) for k, item in enumerate(hist_min)] # (index, hist_min[index])
        hist_min_pair = sorted(hist_min_pair, cmp=lambda x,y:cmp(x[1],y[1])) # sort this array
        h_count = 0
        while count < number_negative_samples and h_count < len(hist_min):
            if hist_min_pair[h_count][1] == -1:
                h_count += 1
                continue

            center = hist_min_pair[h_count][0]
            h_count += 1
            if is_valid_segment(seg_positions, center, threshold=8):
                count += 1
                affine_src = np.array([[center + 1 - crop_width/2, 0],[center + crop_width/2, 0],[center + crop_width/2, crop_width - 1]],dtype=np.float32)
                affine_mat = cv.getAffineTransform(affine_src, affine_dst)
                crop_image = cv.warpAffine(image, affine_mat, dsize=(crop_width, crop_width))
                crop_image_seq.append(crop_image)
                crop_label_seq.append([1, 0])

        # Random crop for negative samples
        while count < number_negative_samples:
            center = random.randint(0, len(label_seq[i]) - 1)
            if is_valid_segment(seg_positions, center, threshold=10):
                count += 1
                affine_src = np.array([[center + 1 - crop_width/2, 0],[center + crop_width/2, 0],[center + crop_width/2, crop_width - 1]],dtype=np.float32)
                affine_mat = cv.getAffineTransform(affine_src, affine_dst)
                crop_image = cv.warpAffine(image, affine_mat, dsize=(crop_width, crop_width))
                crop_image_seq.append(crop_image)
                crop_label_seq.append([1, 0])
                
    return np.reshape(crop_image_seq, [-1, 32, 32, 1]), crop_label_seq

# test crop image
'''
loader = Loader('../psl_data/seg_cnn/traindata_total', ['data_3'], batch_size=100000)
x,y,_,_ = loader.next_batch()
crop_image_list, crop_label_list = crop_image(x * 255.,y,32)
temp_save_dir = '../psl_data/seg_cnn/temp'
if not os.path.exists(temp_save_dir):
    os.mkdir(temp_save_dir)

for i,img in enumerate(crop_image_list):
    name = "{}_{}.bin.png".format(i, 0 if crop_label_list[i][0] == 1 else 1)
    cv.imwrite(os.path.join(temp_save_dir, name), img)
'''
# finish test crop image 

'''
Process space segmentatoin and save results
'''
def process_space_segment(output_seq, pooling_size=1, fg=2, bg=0, min_width=0):
    '''
    dealing with space segmentation, which is a former step for
    character segmentation

    output_seq: 1-D array
    return value: list[list], each item indicate a word's left side
                    and right side
    '''
    N = len(output_seq)
    
    if N < 2:
        return []

    # find space segmentation position
    word_pos = []
    left = 0
    right = 0

    while right < N:
        while left < N and output_seq[left] == fg:
            left += 1

        right = left
        while right < N and output_seq[right] == bg:
            right += 1
        
        if right - left > min_width:
            word_pos.append([pooling_size * left, pooling_size * (right - 1)])

        left = right

    return word_pos

def seg_image(word_pos, image, image_normalize=1.0, is_save=False, image_base_name='', output_dir=''):
    height, width = image.shape[:2]
    image = image * image_normalize
    top = 0  #y
    bottom = height - 1
    words = []

    for i, item in enumerate(word_pos):
        left = item[0]
        right = item[1]
        
        w = int(right - left)
        affine_dst = np.array([[0,0],[w,0],[w,bottom]], dtype=np.float32)
        affine_src = np.array([[left,top],[right,top],[right,bottom]], dtype=np.float32)
        affine_mat = cv.getAffineTransform(affine_src, affine_dst)
        word_image = cv.warpAffine(image, affine_mat, dsize=(w+1, height))

        if is_save:
            # save image
            image_name = "{}_{}.bin.png".format(image_base_name, i)
            cv.imwrite(os.path.join(output_dir, image_name), word_image)
        else:
            words.append(word_image)
            
    return words

def word_segmentation(image_list, argmax_outputs, pooling_size, image_normalize=1.0, is_save=False,output_dir='', batch_number=0):
    '''
    Process word segmentation

    image_list: list, each item is a image mat
    argmax_outputs: list, each item is an array
    output_dir: word segmentation output direction
    batch_number: int, batch size
    pooling_size: int, total pooling size

    return value: None
    '''
    words = []
    for i, image in enumerate(image_list):
        word_pos = process_space_segment(argmax_outputs[i], pooling_size=pooling_size, min_width=2)
        if is_save:
            seg_image(word_pos, image, image_normalize=image_normalize,is_save=True,
                image_base_name='batch_{}_line_{}'.format(batch_number, i), 
                output_dir=output_dir)
        else:
            words += seg_image(word_pos, image, image_normalize=image_normalize,is_save=False)

    return words
        
'''
Process segmentatoin networks' output and save results
'''
def process_cnn_segment(output_seq, pooling_size=1, fg=1, bg=0):
    '''
    dealing with single output sequence 

    output_seq: 1-D array
    return value: 1-D array, segmentation position
            zero-index, not including left and right side
    '''
    N = len(output_seq)

    if N < 2:
        return []

    seg_pos = []
    # removing right side noise
    right = N - 1
    count = right

    while count >= 0 and output_seq[count] == fg:
        count -= 1
    right = count

    # removing left side noise
    count = 0

    while count <= right and output_seq[count] == fg:
        count += 1

    # merging segmentation points
    begin = -1
    end = -1
    while count <= right:
        if begin < 0:
            if output_seq[count] == bg:
                count += 1
            else:
                begin = count
                count += 1
        else:
            if output_seq[count] == bg:
                end = count - 1
                seg_pos.append(pooling_size * int((begin + end) / 2))
                begin = -1
                end = -1
                count += 1
            else:
                count += 1

    return seg_pos

def crop_result_image(image, seg_pos, padding=255, image_normalize=1.0):
    '''
    crop image refering to seg_pos, alse see seg_image() 
    function in utils.py

    image: input image
    seg_pos: segmentation position, list, zero-index
    padding: parameter for opencv
    return value: list, a list of cropped images
    '''
    height, width = image.shape[:2]
    image = image * image_normalize

    # append right side
    seg_pos.append(width - 1)

    top = 0
    left = 0
    bottom = height - 1
    crop_image_list = []

    for i, pos in enumerate(seg_pos):
        right = pos
        left = 0 if i == 0 else seg_pos[i - 1]
        w = int(right - left)

        affine_dst = np.array([[0,0], [w,0], [w, bottom]], dtype=np.float32)
        affine_src = np.array([[left,top], [right,top], [right,bottom]], dtype=np.float32)
        affine_mat = cv.getAffineTransform(affine_src, affine_dst)
        crop_image = cv.warpAffine(image, affine_mat, dsize=(w+1, height))
        crop_image_list.append(crop_image)

    return crop_image_list

def crop_and_save(image_list, argmax_outputs, pooling_size, image_normalize=1.0,is_save=False,output_dir='', batch_number=0):
    '''
    crop images and save cropped images

    image_list: list, each item is a image mat
    argmax_outputs: list, each item is an array whose width is 
                the same as image's width
    output_dir: cropped images output direction
    batch_number: int

    return value: None
    '''
    ans = []
    for i, image in enumerate(image_list):
        seg_pos = process_cnn_segment(argmax_outputs[i], pooling_size=pooling_size)
        crop_image_list = crop_result_image(image, seg_pos, image_normalize=image_normalize)
        # save image
        if is_save:
            for j, crop_image in enumerate(crop_image_list):
                name = "{}_{}_{}.bin.png".format(batch_number, i, j)
                cv.imwrite(os.path.join(output_dir, name), crop_image)
        else:
            ans.append(crop_image_list)

    return ans

'''
add noise to image(numpy array):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    noise = np.random.normal(mean, stddev, (img_height, img_width)
    image += noise
'''

'''
ROC
'''
def match(truth_seq, forecast_seq, neighborhood):
    '''
    Match forecase sequence to groundtruth sequence using greedy algorithm

    Inputs:
    truth_seq: list, in general it is a 0-1 array
    forecast_seq: list, floating points between 0-1
    neighborhood: int, searching possible matching within neighborhood

    Outputs:
    match_seq: list, each item indicate a horizontal position in original image,
    if 0, this position is not matched
    if a floating point between 0-1, this position is matched
    match_prob: list, only store matched points' probability
    '''
    # some check step
    truth_len = len(truth_seq)
    forecast_len = len(forecast_seq)
    if truth_len != forecast_len:
        raise ValueError('groundtruth sequence and forecast sequence do not have equal length: {} : {}'.format(truth_len, forecast_len))
    
    match_seq = [[x, False] for x in forecast_seq]
    match_line_number = 0
    for i, item in enumerate(truth_seq):
        if item == 1:
            left = max(0, i - neighborhood)
            right = min(truth_len - 1, i + neighborhood)

            max_prob = 0
            match_index = -1
            for j in range(left, right + 1):
                if (not match_seq[j][1]) and forecast_seq[j] >= max_prob:
                    max_prob = forecast_seq[j]
                    match_index = j

            '''
            # greedy algorithm may face some problem, but its probability is very small
            if match_index == -1:
                raise ValueError('Can not match sequence under current neighborhood!')
            '''
            if match_index > -1:
                match_seq[match_index][1] = True
                match_line_number += 1

    return match_seq, match_line_number

def single_roc_line(match_seq, match_line_number, truth_line_number):
    '''
    Calculate the ROC line, total complexity is O(N + M + KlogK), "N" is 
    groundtruth line number, "M" is the forecast line number, "K" is the matched
    line number
    Fomula: recall = matched line number / groundtruth line number

    Inputs:
    match_prob: 
    '''
    match_seq = sorted(match_seq, key=lambda d:d[0]) # sort the matched points' probability in ascending order
    match_seq_len = len(match_seq)
    roc_line = [] # item: [greater than threshold line number, matched line number, threshold]

    left = 0
    right = 0
    while left < match_seq_len:
        while right < match_seq_len and match_seq[right][0] == match_seq[left][0]:
            if match_seq[right][1]:
                match_line_number -= 1
            right += 1

        roc_line.append([match_seq_len-right+1, (match_line_number+0.0)/truth_line_number, match_seq[left][0]])

        left = right

    return roc_line

def ROC(truth_seqs, forecast_seqs, neighborhood=4):
    # some check
    if len(truth_seqs) != len(forecast_seqs):
        raise ValueError("truth sequence and forecast sequence \
                do not have same length {}:{}".format(len(truth_seqs), len(forecast_seqs)))

    match_seq = []
    match_line_number = 0
    truth_line_number = 0
    for i, patch in enumerate(truth_seqs):
        for j, truth_seq in enumerate(patch):
            truth_line_number += truth_seq.count(1)
            batch_match_seq, batch_match_line_number = \
                    match(truth_seq, forecast_seqs[i][j], neighborhood)

            match_seq += batch_match_seq
            match_line_number += batch_match_line_number

    print match_line_number, truth_line_number
    roc_line = single_roc_line(match_seq, match_line_number, truth_line_number)

    return roc_line


'''
# test my ROC process
truth_seq = [[0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,1,0],
            [0,0,0,0,1,1,1,0,0,1,0,1,0,1,0,0,0,0,0,0]]
forecast_seq = [[random.uniform(0,1) for i in xrange(20)],
        [random.uniform(0,1) for j in xrange(20)]]

total_roc = ROC(truth_seq, forecast_seq)
print total_roc
'''
