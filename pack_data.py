import os
import Image
import pickle
import numpy as np

def read_file(filename, datadir, outdir, resize = False, newsize = 1):
    image_list = []
    label_list = []
    out_filename = filename.split('/')[-1]
    out_filename = out_filename.split('.')[0]

    with open(filename,'r') as f:
        for name in f:
            if name[-1] == '\n':
                name = name[:-1]
            image = Image.open(os.path.join(datadir, name + '.bin.png'))
            if resize:
                dims = (int(round(newsize * image.size[0] / image.size[1])), newsize)
                image = image.resize(dims)
            image = np.asarray(image, dtype=np.uint32)
            # change to black back ground
            image = 255 - image
            image_list.append(image)

            with open(os.path.join(datadir, name + '.txt')) as lf:
                label = lf.readline()
                if label[-1] == '\n':
                    label = label[:-1]
                mapped_label = []
                for w in label:
                    mapped_label.append(ord(w))
            label_list.append(mapped_label)

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

def make_readfile(datadir, outputdir):
    folders = os.listdir(datadir)
    base_filename = 'data_'
    filesize = 500
    count = 0
    file_list = []
    name_list = []
    for folder in folders:
        name_list += get_all_filename(os.path.join(datadir, folder))
    cnt = 0
    f = open(os.path.join(outputdir, base_filename + str(count) + '.txt'), 'w')
    file_list.append(os.path.join(outputdir, base_filename + str(count) + '.txt'))
    for item in name_list:
        f.write(item)
        f.write('\n')
        cnt += 1
        if cnt % filesize == 0:
            f.close()
            count += 1
            f = open(os.path.join(outputdir, base_filename + str(count) + '.txt'), 'w')
            file_list.append(os.path.join(outputdir, base_filename + str(count) + '.txt'))
            cnt = 0
    if not f.closed:
        f.close()
    return file_list

def bundle_data(file_list, datadir, outdir):
    for f in file_list:
        read_file(f, datadir, outdir, resize=1, newsize=32)

if __name__ == '__main__':
    datadir = '/home/jia/psl/tf_rnn/psl_data/244Images/train_sepf244'
    outputdir = '/home/jia/psl/tf_rnn/psl_data/244Images/trainfile'
    outdir = '/home/jia/psl/tf_rnn/psl_data/244Images/traindata'
    file_list = make_readfile(datadir, outputdir)
    bundle_data(file_list, datadir, outdir)

