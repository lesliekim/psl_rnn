# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:54:51 2015

@author: rabbit
"""

import os
import random
import math
import shutil
import string

def get_all_images(folder): # get all images in one folder
    assert os.path.exists(folder)
    assert os.path.isdir(folder)
    _list = os.listdir(folder)
    image_list = []
    for i in _list:
        if i[len(i)-3:] == "jpg" or i[len(i)-3:] == "png":
            image_list.append(folder + '/' + i)
    return image_list
   
def write_all_files(folders, main_path, output_file_name):
    f = open(output_file_name,'w')
    for folder in folders:
        images = get_all_images(main_path + '/' + folder)
        for image in images:
            f.write(image)
            f.write('\n')
    f.close()
    
def gen_train_test(folders, main_path, train_file_name, 
                   test_file_name, test_p):

    train_f = open(train_file_name, 'w')
    test_f = open(test_file_name, 'w')
    for folder in folders:
        images = get_all_images(main_path + '/' + folder)
        random.shuffle(images)
        sep_index = int(math.floor(len(images) * test_p))
        
        for image in images[0:sep_index]:
            test_f.write(image)
            test_f.write('\n')
        
        for image in images[sep_index:]:
            train_f.write(image)
            train_f.write('\n')
            
    test_f.close()
    train_f.close()
    
def _copy_file(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    #if os.path.exists(src) and  os.path.isdir(dst):
    shutil.copy(src, dst)
        
def copy_files(src_file, dst_path):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        
    with open(src_file, 'r') as f:
        for line in f.readlines():
            line = string.strip(line)
            subfolder = line.split('/')[-2]
            _copy_file(line, dst_path + '/' + subfolder)
            _copy_file(line[:-7] + 'txt', dst_path + '/' +subfolder)
            
    
if __name__ == "__main__":
    main_path = "/home/jia/psl/tf_rnn/psl_data/gulliver/gulliver_out"
    train_dst = "/home/jia/psl/tf_rnn/psl_data/gulliver/gulliver_train"
    test_dst = "/home/jia/psl/tf_rnn/psl_data/gulliver/gulliver_test"
    train_file_name = "/home/jia/psl/tf_rnn/psl_data/gulliver/train.txt"
    test_file_name = "/home/jia/psl/tf_rnn/psl_data/gulliver/test.txt"
    folders = os.listdir(main_path)
    gen_train_test(folders, main_path, train_file_name, test_file_name, 0.2)
    copy_files(train_file_name, train_dst)
    copy_files(test_file_name, test_dst)

        
