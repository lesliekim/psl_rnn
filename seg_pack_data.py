import os
import Image
import pickle
import shutil
import numpy as np
import re
import seg_param as param

label_suffix = param.label_suffix
image_suffix = param.image_suffix
file_size = param.file_size
multi_label = param.multi_label

def read_file(filename, outdir, resize=False, newsize=1, multi_label=multi_label):
    image_list = []
    label_list = []
    out_filename = filename.split('/')[-1].split('.')[0]

    with open(filename,'r') as f:
        for name in f:
            name = name.strip()
            
            image = Image.open(name + image_suffix)
            image = image.convert('L')
            original_image_width = image.size[0]
            original_image_height = image.size[1]

            if not multi_label:
                with open(name + label_suffix) as lf:
                    seg_pos = [float(x.strip()) for x in lf]

                if resize:
                    dims = (int(round(newsize * original_image_width / original_image_height)), newsize)
                    image = image.resize(dims)
                    seg_pos_resize = [int(round(x * newsize / original_image_height)) for x in seg_pos]
                    seg_pos = seg_pos_resize

                # make label
                label = [0] * image.size[0] # width = image.size[0]
                for pos in seg_pos:
                    label[pos - 1] = 1

            elif multi_label == 1:
                with open(name + label_suffix) as lf:
                    seg_pos = [x.strip().split(' ') for x in lf]

                if resize:
                    dims = (int(round(float(newsize) * original_image_width / original_image_height)), newsize)
                    image = image.resize(dims)
                    seg_pos_resize = [[int(round(float(newsize) * int(x[0]) / original_image_height)), int(x[1])] for x in seg_pos]
                    seg_pos = seg_pos_resize

                # make label
                label = [0] * image.size[0] # width = image.size[0]
                for item in seg_pos:
                    label[min(item[0] - 1, image.size[0] - 1)] = item[1]

            else:
                with open(name + label_suffix) as lf:
                    seg_pos = [x.strip().split(' ') for x in lf]

                if resize:
                    dims = (int(round(float(newsize) * original_image_width / original_image_height)), newsize)
                    image = image.resize(dims)
                    seg_pos_resize = [[int(round(float(newsize) * int(x[0]) / original_image_height)), int(x[1])] for x in seg_pos]
                    seg_pos = seg_pos_resize

                # make label
                label = [0] * image.size[0]
                seg_pos_len = len(seg_pos)
                pos_count = 0
                while pos_count < seg_pos_len:
                    if seg_pos[pos_count][1] == 2:
                        left = seg_pos[pos_count][0] - 1
                        right = (seg_pos[pos_count+1][0] - 1) if pos_count < seg_pos_len-1 else (image.size[0]-1)
                        label[left:right+1] = [2] * (right + 1 - left)
                        pos_count += 1
                    pos_count += 1
                if len(label) > image.size[0]:
                    print "long " + name
                    print len(label)
                    print image.size

            label_list.append(label)

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
    n_height = param.height
    is_resize = param.resize

    for f in file_list:
        read_file(f, outdir, resize=is_resize, newsize=n_height)

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
    datadir = ['/home/jia/psl/tf_rnn/psl_data/seg_cnn/train_org_for_space_3']
    readfile_outdir = '/home/jia/psl/tf_rnn/psl_data/seg_cnn/trainfile_for_space_3'
    data_outdir = '/home/jia/psl/tf_rnn/psl_data/seg_cnn/traindata_for_space_3'
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
