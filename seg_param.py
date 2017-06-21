height=32
resize=1
label_suffix='.pos'
image_suffix='.bin.png'
file_size = 500 #20000
multi_label = 2 # 0: only[0,1]; 1: [0,1,2], '2' is for space; 2: no pos file at all, just for real samples; else: for word space segmentation 
pad_value = 1 # value for padding, it is used for sequence whose length is small than batch_max_length
