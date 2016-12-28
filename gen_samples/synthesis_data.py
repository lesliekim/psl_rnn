# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:29:55 2016
synthesis data from groundtruth
@author: rabbit
"""

import os
import utils
import pygame
import random
import numpy

pygame.init()

text_dir = "D:/master_work/dataset/father_groundtruth"
out_dir = "D:/master_work/dataset/synthesis_data_father"
pos_dir = "D:/master_work/dataset/synthesis_data__father_position"
font_family = ['timesnewroman', 'calibri', 'cambria']
font_sizes = [80, 90, 100]
folders = utils.get_all_folders(text_dir)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    
if not os.path.exists(pos_dir):
    os.mkdir(pos_dir)
    
for folder in folders:
    print('processing folder: {name}'.format(name=os.path.join(text_dir, folder)))
    
    text_files = utils.get_all_files(os.path.join(text_dir, folder), set(['txt']))
    output_folder = os.path.join(out_dir, folder)
    position_folder = os.path.join(pos_dir, folder)
    
    font_type = font_family[random.randint(0, len(font_family)-1)]
    font_size = font_sizes[random.randint(0, len(font_sizes)-1)]
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(position_folder):
        os.mkdir(position_folder)
        
    for text_file in text_files:
        image_name = text_file.split('.')[0] + '.bin.png'
        groundtruth_name = text_file
        segment_position_name = text_file
        
        with open(os.path.join(text_dir, folder, text_file), 'r') as f:
            text = f.read().strip()
            '''
            # to uppercase
            text = text.upper()
            # render squence samples
            image, rect = utils.render_from_text(text, font_type, font_size)
            '''
            # render adhesion samples
            image, seg_pos = utils.adhesion_samples(text, font_type, font_size,
                                                    space_pad=3)
            # segment position is only for adhesion samples
            numpy.savetxt(os.path.join(position_folder, segment_position_name), 
                          seg_pos, fmt="%d") 
            
            pygame.image.save(image, os.path.join(output_folder, image_name))
            
            with open(os.path.join(output_folder, groundtruth_name), 'w') as gf:
                gf.write(text)
                
            
            
            
    



        
    