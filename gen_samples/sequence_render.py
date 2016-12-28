# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 10:56:51 2016

@author: rabbit

read text and render sample
"""
import os
import string
import pygame
import numpy
import utils

'''
current font: times, calibri, cambria
'''
def process_render(text):
    global image_count
    global output_dir
    global font_name
    
    image, seg_pos = utils.adhesion_samples(text, font_type=font_name, 
                                            font_size=40)
    pygame.image.save(image, os.path.join(output_dir, 
                            "{}_{}.bin.png".format(font_name, image_count)))
                            
    numpy.savetxt(os.path.join(output_dir, 
                "{}_{}.txt".format(font_name, image_count)), seg_pos, fmt="%d")                             
    image_count += 1
    
    if image_count % 1000 == 0:
        print("processing {} data...".format(image_count))
        
        
pygame.init()

font_name = 'times'
output_dir = "D:/master_work/dataset/segmentation_data_for_CNN_{}_2".format(font_name)
prefix_punctuation = '-"([{' + "'"
suffix_punctuation = '-")]};,:!%@?' 
number = '0123456789'

image_count = 0

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# render single character samples
for ch1 in string.uppercase + string.lowercase + string.punctuation:
    process_render(ch1)

# two characters 
### uppercase + lowercase 
for ch1 in string.uppercase + prefix_punctuation:
    for ch2 in string.lowercase + suffix_punctuation:
        text = ''.join([ch1, ch2])
        process_render(text)

### lowercase + lowercase        
for ch1 in string.lowercase + prefix_punctuation:
    for ch2 in string.lowercase + suffix_punctuation:
        text = ''.join([ch1, ch2])
        process_render(text)

# three characters
for ch1 in string.uppercase + string.lowercase + prefix_punctuation:
    for ch2 in string.lowercase:
        for ch3 in string.lowercase + suffix_punctuation:
            text = ''.join([ch1,ch2,ch3])
            process_render(text)
            
# fallibility case 
### number
for ch1 in number:
    for ch2 in number:
        for ch3 in number:
            text = ''.join([ch1, ch2, ch3])
            process_render(text)

### sticky character
set_1 = 'nNmMsS'
set_2 = 'Ii1lj'
set_3 = 'cCoO0N'
set_4 = 'XxWwVv'
set_5 = 'th'
set_6 = 'qpg'
set_7 = ',):"!@'

for i in xrange(10): # raise weight
    # case 1: set 1 + set 2 + other characters
    for ch1 in set_1:
        for ch2 in set_2:
            for ch3 in string.uppercase + string.lowercase + suffix_punctuation:
                text = ''.join([ch1, ch2, ch3])
                process_render(text)
                
    # case 2: set 3 + set 4 + other characters
    for ch1 in set_3:
        for ch2 in set_4:
            for ch3 in string.uppercase + string.lowercase + suffix_punctuation:
                text = ''.join([ch1, ch2, ch3])
                process_render(text)        
    
    # case 3: set 5 + set 6 + other characters
    for ch1 in set_5:
        for ch2 in set_6:
            for ch3 in string.uppercase + string.lowercase + suffix_punctuation:
                text = ''.join([ch1, ch2, ch3])
                process_render(text)
                
    # case 4: other characters + set 7
    for ch1 in string.uppercase + string.lowercase:
        for ch2 in set_7:
            text = ''.join([ch1, ch2])
            process_render(text)