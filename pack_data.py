import os
import Image
import pickle
import shutil
import numpy as np
import re
import param

label_suffix = param.label_suffix
image_suffix = param.image_suffix

def read_file(filename, datadir, outdir, resize=False, newsize=1):
    image_list = []
    label_list = []
    out_filename = filename.split('/')[-1]
    out_filename = out_filename.split('.')[0]

    threshold = 90 # throw away too long data(trick, hehe)
    num_classes = 128 # how many labels

    with open(filename,'r') as f:
        for name in f:
            name = name.strip()
            
            is_valid = True
            with open(name + label_suffix) as lf:
                label = lf.readline()
                label = label.strip()
                if len(label) > threshold:
                    is_valid = False
                    print "too long " + str(len(label)) + " " + name
                else:
                    mapped_label = []
                    mylabel = ''
                    for w in label:
                        if ord(w) >= num_classes - 1: # out of basic asic
                            is_valid = False
                        else:
                            mapped_label.append(ord(w))

                    if is_valid:
                        label_list.append(mapped_label)

            if is_valid:
                image = Image.open(name + image_suffix)
                image = image.convert('L')
                if resize:
                    dims = (int(round(newsize * image.size[0] / image.size[1])), newsize)
                    image = image.resize(dims)
                image = np.asarray(image, dtype=np.uint32)
                # change to black back ground
                image = 255 - image
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
    base_filename = 'data_'
    filesize = 500
    file_list = []
    name_list = []
    # get all images
    if has_subfolder:
        folders = os.listdir(datadir)
        for folder in folders:
            name_list += get_all_filename(os.path.join(datadir, folder))
    else:
        name_list = list(set([os.path.join(datadir, x.split('.')[0]) for x in os.listdir(datadir)]))

    cnt = 0
    f = open(os.path.join(outputdir, base_filename + str(readfile_count) + '.txt'), 'w')
    file_list.append(os.path.join(outputdir, base_filename + str(readfile_count) + '.txt'))
    for i,item in enumerate(name_list):
        f.write(item)
        f.write('\n')
        cnt += 1
        if cnt % filesize == 0:
            f.close()
            readfile_count += 1
            if i == len(name_list) - 1:
                break
            f = open(os.path.join(outputdir, base_filename + str(readfile_count) + '.txt'), 'w')
            file_list.append(os.path.join(outputdir, base_filename + str(readfile_count) + '.txt'))
            cnt = 0
    if not f.closed:
        f.close()
    return file_list

def get_readfile(readfile_dir):
    files = os.listdir(readfile_dir)
    file_list = [os.path.join(readfile_dir, f) for f in files]
    return file_list

def bundle_data(file_list, datadir, outdir):
    n_height = param.height
    is_resize = param.resize

    for f in file_list:
        read_file(f, datadir, outdir, resize=is_resize, newsize=n_height)

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
            dst_file = os.path.join(dst_path, f)
            shutil.copyfile(src_file, dst_file)

#movefile('/home/jia/psl/tf_rnn/psl_data/gulliver_groundtruth','/home/jia/psl/tf_rnn/psl_data/gulliver_out')

if __name__ == '__main__':
    datadir = ['/home/jia/psl/tf_rnn/psl_data/English_card_hard/namecard_binary'] 
    readfile_outdir = '/home/jia/psl/tf_rnn/psl_data/English_card_hard/namecard_binary_trainfile'
    data_outdir = '/home/jia/psl/tf_rnn/psl_data/English_card_hard/namecard_binary_traindata'
    has_readfile = False
    has_subfolder = False # if datadir has subfolders and images are in different subfolders, 
                            # set this variable to be True
    
    if not os.path.exists(data_outdir):
        os.mkdir(data_outdir)
    if not os.path.exists(readfile_outdir):
        os.mkdir(readfile_outdir)

    if has_readfile:
        file_list = get_readfile(readfile_outdir)
    else:
        if type(datadir) == list:
            file_list = []
            for d in datadir:
                file_list.extend(make_readfile(d, readfile_outdir, has_subfolder))
        else:
            file_list = make_readfile(datadir, readfile_outdir, has_subfolder)

    bundle_data(file_list, datadir, data_outdir)
