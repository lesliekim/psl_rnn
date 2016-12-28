# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 10:58:35 2016

@author: rabbit

common tool functions
"""
import os
import pygame
import pygame.freetype as freetype


pygame.init()
freetype.init()

def get_all_folders(path, add_path=False):
    folders = os.listdir(path)
    ans = []
    for folder in folders:
        if os.path.isdir(os.path.join(path, folder)):
            if add_path:
                ans.append(os.path.join(path, folder))
            else:
                ans.append(folder)
    return ans
    
def get_all_files(folder_dir, suffix=set(), add_path=False): 
    files = os.listdir(folder_dir)
    ans = []
    for f in files:
        if f.split('.')[-1] in suffix:
            if add_path:
                ans.append(os.path.join(folder_dir, f))
            else:
                ans.append(f)
    return ans
        
def render_from_text(text, font_type, font_size=16, fg=(0,0,0), 
                     bg=(255,255,255)):
    font = freetype.Font(font_type, font_size)
    image, rect = font.render(text, fg, bg)
    
    return image, rect

def adhesion_samples(text, font_type='cambria', font_size=16, 
                     fg=(0,0,0), bg=(255,255,255), space_pad=0):
    font = freetype.SysFont(font_type, font_size)
    image, rect = font.render(text, fg, bg)

    char_rect = []
    for c in text:
        c_rect = font.get_rect(c)
        char_rect.append(c_rect)
      
    img_width = sum(x[2] for x in char_rect) + 1 + space_pad * (len(text) - 1)
    img_height = rect[3]
    box_height = max(x[1] for x in char_rect)
    img = pygame.Surface((img_width, img_height))
    img.fill(bg)
    
    left = 1
    seg_pos = []
    for i,c in enumerate(text):
        top = box_height - char_rect[i][1]
        font.render_to(img, (left, top), c, fg, bg)
        left += char_rect[i][2] + space_pad
        seg_pos.append(left)
     
    seg_pos.pop() # the last position should not be added to segmentation position
        
    return img, seg_pos

        
       