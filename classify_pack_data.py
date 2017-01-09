import os
import cv2 as cv
import pickle
import shutil
import numpy as np
import re
import classify_param as param

label_suffix = param.label_suffix
image_suffix = param.image_suffix
file_size = param.file_size
resize_height = param.resize_height
resize_width = param.resize_width

def pad_to_square(image):
    h = image.shape[0]
    w = image.shape[1]
    if h > w:
        p_left = p_right = (h + 4 - w) / 2
        p_top = p_bottom = 2
    else:
        p_left = p_right = 2
        p_top = p_bottom = (w + 4 - h) / 2
    image = cv.copyMakeBorder(image, p_top, p_bottom, p_left, p_right, 
            borderType=cv.BORDER_CONSTANT, value=0)

    if resize:
        original_height = image.shape[0]
        original_width = image.shape[1]
        affine_dst = np.array([[0,0],[resize_width-1,0],[resize_width-1,resize_height-1],], dtype=np.float32)
        affine_src = np.array([[0,0],[original_width-1,0],[original_width-1,original_height-1]], dtype=np.float32)

        affine_mat = cv.getAffineTransform(affine_src, affine_dst)
        image = cv.warpAffine(image, affine_mat, dsize=(resize_width, resize_height))

    image = np.asarray(image, dtype=np.uint32)
    return image

def read_file(filename, outdir, resize=False):
    image_list = []
    label_list = []
    out_filename = filename.split('/')[-1].split('.')[0]

    with open(filename,'r') as f:
        for name in f:
            name = name.strip()
            
            with open(name + label_suffix) as lf:
                label = lf.readline().strip()

            image = cv.imread(name + image_suffix, cv.CV_LOAD_IMAGE_GRAYSCALE)
            image = 255 - image
            
            # pad to square
            h = image.shape[0]
            w = image.shape[1]
            if h > w:
                p_left = p_right = (h + 4 - w) / 2
                p_top = p_bottom = 2
            else:
                p_left = p_right = 2
                p_top = p_bottom = (w + 4 - h) / 2
            image = cv.copyMakeBorder(image, p_top, p_bottom, p_left, p_right, 
                    borderType=cv.BORDER_CONSTANT, value=0)

            if resize:
                original_height = image.shape[0]
                original_width = image.shape[1]
                affine_dst = np.array([[0,0],[resize_width-1,0],[resize_width-1,resize_height-1],], dtype=np.float32)
                affine_src = np.array([[0,0],[original_width-1,0],[original_width-1,original_height-1]], dtype=np.float32)

                affine_mat = cv.getAffineTransform(affine_src, affine_dst)
                image = cv.warpAffine(image, affine_mat, dsize=(resize_width, resize_height))

            label_list.append(int(label))
            image = np.asarray(image, dtype=np.uint32)
            # change to black back ground
            image_list.append(image)

    image_outfile = os.path.join(outdir, out_filename + '_image.p')
    print('Writing to: ', image_outfile)
    outfile = open(image_outfile, 'wb')
    pickle.dump(image_list, outfile)
    outfile.close()

    label_outfile = os.path.join(outdir, out_filename + '_label.p')
    print('Writing to: ', label_outfile)
    outfile = open(label_outfile, 'wb')
    pickle.dump(label_list, outfile)
    outfile.close()

def get_all_filename(pathname):
    filename_list = []
    names = os.listdir(pathname)
    for name in names:
        char = name.split('.')
        if char[1] == 'txt':
            filename_list.append(os.path.join(pathname, char[0]))

    return filename_list

readfile_count = 0
def make_readfile(datadir, outputdir, has_subfolder=False):
    '''
    datadir: data direction
    outputdir:separating files direction.separate data into small set for pickle,
    '''
    global readfile_count
    if not has_subfolder:
        name_list = list(set([os.path.join(datadir, x.split('.')[0]) for x in os.listdir(datadir)]))
    else:
        name_list = []
        folders = os.listdir(datadir)
        for folder in folders:
            name_list += get_all_filename(os.path.join(datadir, folder))
        
    base_filename = 'data_'
    filesize = file_size
    file_list = []
    cnt = 0

    filename = os.path.join(outputdir, base_filename + str(readfile_count) + '.txt')
    f = open(filename, 'w')
    file_list.append(filename)
    for item in name_list:
        f.write(os.path.join(datadir, item))
        f.write('\n')
        cnt += 1
        if cnt % filesize == 0:
            f.close()
            readfile_count += 1

            filename = os.path.join(outputdir, base_filename + str(readfile_count) + '.txt')
            f = open(filename, 'w')
            file_list.append(filename)
            cnt = 0
    if not f.closed:
        f.close()
    return file_list

def get_readfile(readfile_dir):
    files = os.listdir(readfile_dir)
    file_list = [os.path.join(readfile_dir, f) for f in files]
    return file_list

def bundle_data(file_list, outdir):
    is_resize = param.resize

    for f in file_list:
        read_file(f, outdir, resize=is_resize)

def movefile(src_dir, dst_dir):
    assert os.path.exists(src_dir)
    assert os.path.exists(dst_dir)
    src_folders = os.listdir(src_dir)
    for folder in src_folders:
        src_path = os.path.join(src_dir, folder)
        dst_path = os.path.join(dst_dir, folder)
        print('Moving files from ' + src_path + ' to ' + dst_path)
        files = os.listdir(src_path)
        for f in files:
            src_file = os.path.join(src_path, f)
            dst_file = os.path.join(dst_path, f[:-3]+'std')
            shutil.copyfile(src_file, dst_file)

#movefile('/home/jia/psl/tf_rnn/psl_data/father/synthesis_data_father_position_withspace',
#'/home/jia/psl/tf_rnn/psl_data/father/synthesis_data_father_withspace')
if __name__ == '__main__':
    datadir = ['/home/jia/psl/tf_rnn/psl_data/classify_cnn/9font_4style_0destort']
    readfile_outdir = '/home/jia/psl/tf_rnn/psl_data/classify_cnn/9font_4style_0destort_trainfile'
    data_outdir = '/home/jia/psl/tf_rnn/psl_data/classify_cnn/9font_4style_0destort_traindata'
    has_readfile = False
    has_subfolder = False
    
    if not os.path.exists(data_outdir):
        os.mkdir(data_outdir)
    if not has_readfile and not os.path.exists(readfile_outdir):
        os.mkdir(readfile_outdir)

    # file_list should have the full absolute path except suffix
    if has_readfile:
        file_list = get_readfile(readfile_outdir)
    else:
        if type(datadir) == list:
            file_list = []
            for d in datadir:
                file_list.extend(make_readfile(d, readfile_outdir, has_subfolder))
        else:
            file_list = make_readfile(datadir, readfile_outdir, has_subfolder)
    print "Begin bundle data"
    bundle_data(file_list, data_outdir)
