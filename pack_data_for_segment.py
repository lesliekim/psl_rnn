import os
import Image
import pickle
import shutil
import numpy as np
import re
import seg_param as param


def read_file(filename, datadir, outdir, resize=False, newsize=1):
    image_list = []
    label_list = []
    out_filename = filename.split('/')[-1].split('.')[0]

    with open(filename,'r') as f:
        for name in f:
            name = name.strip()
            
            with open(os.path.join(datadir, name + '.txt')) as lf:
                seg_pos = [float(x.strip()) for x in lf]

            image = Image.open(os.path.join(datadir, name + '.bin.png'))
            image = image.convert('L')
            if resize:
                dims = (int(round(newsize * image.size[0] / image.size[1])), newsize)
                image = image.resize(dims)
                seg_pos = [int(round(x * newsize / image.size[1])) for x in seg_pos]

            # make label
            label = [0] * image.size[0] # width = image.size[0]
            for pos in seg_pos:
                label[pos] = 1
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
def make_readfile(datadir, outputdir):
    '''
    datadir: data direction
    outputdir:separating files direction.separate data into small set for pickle,
    '''
    global readfile_count
    name_list = set([x.split('.')[0] for x in os.listdir(datadir)])
    base_filename = 'data_'
    filesize = 10000
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
    datadir = ['/home/jia/psl/tf_rnn/psl_data/seg_data/train_org']
    readfile_outdir = '/home/jia/psl/tf_rnn/psl_data/seg_data/trainfile'
    data_outdir = '/home/jia/psl/tf_rnn/psl_data/seg_data/traindata'
    has_readfile = False
    
    if not os.path.exists(data_outdir):
        os.mkdir(data_outdir)
    if not has_readfile and not os.path.exists(readfile_outdir):
        os.mkdir(readfile_outdir)

    if has_readfile:
        file_list = get_readfile(readfile_outdir)
    else:
        if type(datadir) == list:
            file_list = []
            for d in datadir:
                file_list.extend(make_readfile(d, readfile_outdir))
        else:
            file_list = make_readfile(datadir, readfile_outdir)

    bundle_data(file_list, datadir, data_outdir)
