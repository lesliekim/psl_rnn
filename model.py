import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc  # ctc no more in contrib
from tensorflow.contrib import grid_rnn
import param
import seg_param
import math
import utils
import numpy as np

height = param.height
seg_height = seg_param.height

class CnnRnnModel(object):
    def __init__(self, batch_size=1):
        # Network configs
        momentum = 0.9
        learning_rate = 1e-4
        model_type = 'lstm' #'gru'
        image_height = height #48
        # Accounting the 0th indice + blank label = 121 characters
        num_classes = 128 #121
        num_hidden_1 = 50
        num_hidden_2 = 100

        # Placeholders
        self.inputs = tf.placeholder(tf.float32, [None, None, image_height, 1], name="inputs")
        # SparseTensor required by ctc_loss op.
        self.targets = tf.sparse_placeholder(tf.int32, name="targets")
        # 1d array of size [batch_size]
        self.seq_len = tf.placeholder(tf.int64, [None], name="seq_len")

        # Conv layer
        shape = tf.shape(self.inputs)
        batch_s, max_timesteps = shape[0], shape[1]
        inputs = (self.inputs - 0.1) / 0.3
        ksize_conv1 = 5
        stride_conv1 = 2
        channel_conv1 = 16

        W_conv1 = tf.Variable(tf.truncated_normal([ksize_conv1, ksize_conv1, 1, channel_conv1], stddev=0.2))
        b_conv1 = tf.Variable(tf.truncated_normal([channel_conv1], mean=0, stddev=0))
        y_conv1 = tf.nn.conv2d(inputs, W_conv1, strides=[1, stride_conv1, stride_conv1, 1], padding='SAME')
        h_conv1 = tf.nn.relu(y_conv1 + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        seq_len = ((self.seq_len - 1) / stride_conv1 + 1 + 1) / 2
        num_features = (((image_height - 1) / stride_conv1 + 1 + 1) / 2) * channel_conv1

        h_pool1 = tf.reshape(h_pool1, [batch_s, -1, num_features])

        # RNN layer
        additional_cell_args = {}
        if model_type == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif model_type == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif model_type == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
            additional_cell_args.update({'state_is_tuple': True})
        elif model_type == 'lstm peep':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
            additional_cell_args.update({'state_is_tuple': True, 'use_peepholes': True})
        elif model_type == 'gridlstm':
            cell_fn = grid_rnn.Grid2LSTMCell
            additional_cell_args.update({'use_peepholes': True, 'forget_bias': 1.0})
        elif model_type == 'gridgru':
            cell_fn = grid_rnn.Grid2GRUCell
        else:
            raise Exception("model type not supported: {}".format(model_type))

        rnn_fw_1 = cell_fn(num_hidden_1, **additional_cell_args)
        rnn_bw_1 = cell_fn(num_hidden_1, **additional_cell_args)
        rnn_fw_2 = cell_fn(num_hidden_2, **additional_cell_args)
        rnn_bw_2 = cell_fn(num_hidden_2, **additional_cell_args)

        with tf.variable_scope('layer1'):
            outputs_1, _ = tf.nn.bidirectional_dynamic_rnn(
                rnn_fw_1, rnn_bw_1, h_pool1, seq_len,
                dtype=tf.float32, parallel_iterations=batch_size)
            outputs_1 = tf.concat(2, outputs_1)
        with tf.variable_scope('layer2'):
            outputs_2, _ = tf.nn.bidirectional_dynamic_rnn(
                rnn_fw_2, rnn_bw_2, outputs_1, seq_len,
                dtype=tf.float32, parallel_iterations=batch_size)
            outputs = tf.concat(2, outputs_2)

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden_2 * 2])  # bi_directional

        # Truncated normal with mean 0 and stdev=0.1
        W = tf.Variable(tf.truncated_normal([num_hidden_2 * 2, num_classes], stddev=0.01))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b
        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])
        self.logits_prob = tf.nn.softmax(logits, dim=-1)

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

        loss = ctc.ctc_loss(logits, self.targets, tf.cast(seq_len, dtype='int32'))
        self.cost = tf.reduce_mean(loss)

        self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.cost)

        # faster
        # decoded, log_prob = ctc.ctc_greedy_decoder(logits, tf.cast(seq_len, dtype='int32'))

        # slower but better
        self.decoded, self.log_prob = ctc.ctc_beam_search_decoder(logits, tf.cast(seq_len, dtype='int32'))

        # Accuracy: label error rate
        self.err = tf.reduce_sum(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets, normalize=False))

        self.saver = tf.train.Saver()


class BiRnnModel(object):
    def __init__(self, batch_size=1, keep_prob_1=1.0, keep_prob_2=1.0, keep_prob_3=1.0):
        # Network configs
        momentum = 0.9
        learning_rate = 1e-4
        model_type = 'lstm'
        image_height = height #48
        # Accounting the 0th indice + blank label = 121 characters
        num_classes = 128 #121
        num_hidden_1 = 50
        num_hidden_2 = 100
        num_hidden_3 = 200

        # Has size [batch_size, max_stepsize, num_features], but the
        # batch_size and max_stepsize can vary along each step
        self.inputs = tf.placeholder(tf.float32, [None, None, image_height, 1], name="inputs")
        inputs = tf.reshape(self.inputs, [tf.shape(self.inputs)[0], -1, image_height])
        # TODO(rabbit): modity the normalization
        inputs = (inputs - 0.1) / 0.3

        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        self.targets = tf.sparse_placeholder(tf.int32, name="targets")

        # 1d array of size [batch_size]
        self.seq_len = tf.placeholder(tf.int64, [None], name="seq_len")

        # Defining the cell
        # Can be:

        additional_cell_args = {}
        if model_type == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif model_type == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif model_type == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
            additional_cell_args.update({'state_is_tuple': True})
        elif model_type == 'lstm peep':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
            additional_cell_args.update({'state_is_tuple': True, 'use_peepholes': True})
        elif model_type == 'gridlstm':
            cell_fn = grid_rnn.Grid2LSTMCell
            additional_cell_args.update({'use_peepholes': True, 'forget_bias': 1.0})
        elif model_type == 'gridgru':
            cell_fn = grid_rnn.Grid2GRUCell
        else:
            raise Exception("model type not supported: {}".format(model_type))

        rnn_fw_1 = cell_fn(num_hidden_1, **additional_cell_args)
        rnn_bw_1 = cell_fn(num_hidden_1, **additional_cell_args)
        rnn_fw_2 = cell_fn(num_hidden_2, **additional_cell_args)
        rnn_bw_2 = cell_fn(num_hidden_2, **additional_cell_args)
        rnn_fw_3 = cell_fn(num_hidden_3, **additional_cell_args)
        rnn_bw_3 = cell_fn(num_hidden_3, **additional_cell_args)

        rnn_fw_1 = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_1, input_keep_prob=keep_prob_1)
        rnn_bw_1 = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_1, input_keep_prob=keep_prob_1)
        rnn_fw_2 = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_2, input_keep_prob=keep_prob_2)
        rnn_bw_2 = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_2, input_keep_prob=keep_prob_2)
        rnn_fw_3 = tf.nn.rnn_cell.DropoutWrapper(rnn_fw_3, input_keep_prob=keep_prob_3)
        rnn_bw_3 = tf.nn.rnn_cell.DropoutWrapper(rnn_bw_3, input_keep_prob=keep_prob_3)

        # The second output is the last state and we will no use that
        # outputs, state = tf.nn.rnn(cell, inputs, dtype=tf.float32)

        with tf.variable_scope('layer1'):
            outputs_1, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_1, rnn_bw_1, inputs, self.seq_len,
                                                           dtype=tf.float32, parallel_iterations=batch_size)
            # TODO(rabbit): remove it
            outputs_1 = tf.concat(2, outputs_1)
        with tf.variable_scope('layer2'):
            outputs_2, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_2, rnn_bw_2, outputs_1, self.seq_len,
                                                           dtype=tf.float32, parallel_iterations=batch_size)
            outputs_2 = tf.concat(2, outputs_2)
        with tf.variable_scope('layer3'):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_3, rnn_bw_3, outputs_2, self.seq_len,
                                                         dtype=tf.float32, parallel_iterations=batch_size)
            outputs = tf.concat(2, outputs)

        shape = tf.shape(inputs)
        batch_s, max_timesteps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden_3 * 2])  # bi_directional

        # Truncated normal with mean 0 and stdev=0.1
        W = tf.Variable(tf.truncated_normal([num_hidden_3 * 2, num_classes], stddev=0.01))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        self.logits_prob = tf.nn.softmax(logits, dim=-1)

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

        loss = ctc.ctc_loss(logits, self.targets, tf.cast(self.seq_len, dtype='int32'))
        self.cost = tf.reduce_mean(loss)

        self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.cost)


        self.decoded, self.log_prob = ctc.ctc_beam_search_decoder(logits, tf.cast(self.seq_len, dtype='int32'))


        # Accuracy: label error rate
        self.err = tf.reduce_sum(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets, normalize=False))

        self.saver = tf.train.Saver(max_to_keep=0)


class DeepConv(object):
    '''
    conv_num_list: a list of list
                    [[filter_size, layer_count, ...out_channel, 
                    pool_kernel_height, pool_kernel_width],...]

                    filter_size: int
                    layer_count: int, the number of the same conv layers
                    out_channel: int, the number of out_channel depends on layer_count
                    pool_kernel_height: int
                    pool_kernel_width: int
    if you want to skip pooling, set pooling height and width to zero
    '''
    def __init__(self, conv_num_list):
        self._conv_num_list = conv_num_list

    def deep_conv_layers(self, in_channel, conv):
        for stage_num, stage_param in enumerate(self._conv_num_list):
            stage_name = "stage{}_".format(stage_num)
            filter_size = stage_param[0]
            layer_count = stage_param[1]
            for layer_num in xrange(layer_count):
                out_channel = stage_param[layer_num + 2]
                with tf.variable_scope("{}conv{}".format(stage_name, layer_num)):
                    weights = tf.get_variable("weights",
                            initializer=tf.truncated_normal([filter_size, filter_size,
                                in_channel, out_channel],
                                stddev=math.sqrt(2.0 / (filter_size ** 2 * in_channel))))
                    biases = tf.get_variable("biases",
                            initializer=tf.zeros([out_channel]))
                    conv = tf.nn.conv2d(conv, weights,
                            strides=[1, 1, 1, 1], padding="SAME")
                    conv = tf.nn.bias_add(conv, biases)
                with tf.variable_scope("{}relu{}".format(stage_name, layer_num)):
                    relu = tf.nn.relu(conv)
                in_channel = out_channel
                conv = relu

            if stage_param[2 + layer_count] > 0: # if you want to skip pooling, set pooling height and width to zero
                with tf.variable_scope("{}pool".format(stage_name)):
                    pool_height = stage_param[2 + layer_count]
                    pool_width = stage_param[3 + layer_count]
                    pool = tf.nn.max_pool(conv, ksize=[1, pool_height, pool_width, 1],
                            strides=[1, pool_height, pool_width, 1], padding="SAME")
                conv = pool

        return conv


class SegNet(object):
    def __init__(self, batch_size, is_test=False):
        self.inputs = tf.placeholder(tf.float32, shape=[None, 32, None, 1], name="inputs")
        inputs = self.inputs
        self.targets = tf.placeholder(tf.int64, shape=[None, None], name="targets")
        targets = self.targets

        conv_layer = [[5, 1, 32, 2, 2],[5, 1, 64, 2, 2],[8, 1, 128, 0, 0], [1, 1, 2, 0, 0]]

        deep_conv = DeepConv(conv_layer).deep_conv_layers(1, inputs)
        '''
        # MNIST original network
        with tf.variable_scope("fc"):
            # densely connected layer
            w_fc1 = tf.get_variable("weight",
                    initializer=tf.truncated_normal([-1, 1024], stddev=0.1))
            b_fc1 = tf.get_variable("bias",
                    initializer=tf.constant(0.1, shape=[1024]))

            h_pool2_flat = tf.reshape(deep_conv, [-1, ])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

            if not is_test:
                # dropout
                keep_prob = 0.5
                h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
            # readout layer
            w_fc2 = tf.get_variable("weight_2",
                    initializer=tf.truncated_normal([1024,-1], stddev=0.1))
            b_fc2 = tf.get_variable("bias_2",
                    initializer=tf.constant(0.1, shape=None))
            y_conv = tf.matmul(h_fc1, w_fc2) + b_fc2
        '''
        with tf.variable_scope("avg_pool"):
            pool_height = 8
            pool_width = 1
            pool = tf.nn.avg_pool(deep_conv, ksize=[1, pool_height, pool_width, 1],
                    strides=[1, pool_height, pool_width, 1], padding="SAME")
            deep_conv = pool
        y_conv = deep_conv
        '''
        with tf.variable_scope("label_pool"):
            target_pool = tf.cast(tf.reshape(targets, [batch_size, -1, 2, 1]), tf.float32)
            pool_height = 4
            pool_width = 1
            pool = tf.nn.max_pool(target_pool, ksize=[1, pool_height, pool_width, 1],
                    strides=[1, pool_height, pool_width, 1], padding="SAME")
            targets = tf.cast(tf.reshape(pool, [batch_size, -1, 2]), tf.int64)
        '''
        with tf.variable_scope("label_pool"):
            target_pool = tf.cast(tf.reshape(targets, [batch_size, -1, 1, 1]), tf.float32)
            pool_height = 4
            pool_width = 1
            pool = tf.nn.max_pool(target_pool, ksize=[1, pool_height, pool_width, 1],
                    strides=[1, pool_height, pool_width, 1], padding="SAME")
            targets = tf.cast(pool, dtype=tf.int64)

        if not is_test:
            # loss
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, targets))
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        
        y_softmax = tf.nn.softmax(y_conv, dim=-1)
        self.argmax_outputs = tf.argmax(y_softmax, 2)
        self.argmax_targets = targets

        correct_prediction = tf.equal(self.argmax_outputs, self.argmax_targets)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def RecNet(object):
    def __init__(self, is_test=False):
        self.inputs = tf.placeholder(tf.float32, shape=[], name="inputs")
        inputs = self.inputs
        self.targets = tf.placeholder(tf.float32, shape=[], name="targets")
        targets = self.targets
        
        conv_layer = []
        with tf.variable_scope("fc"):
            pass

class SegRnnNet(object):
    def __init__(self, batch_size, is_test=False):
        img_height = seg_height
        num_hidden_1 = 50

        self.inputs = tf.placeholder(tf.float32, shape=[None, img_height, None, 1], name="inputs")
        inputs = self.inputs
        self.seq_len = tf.placeholder(tf.int64, [None], name="seq_len")
        seq_len = self.seq_len
        self.targets = tf.placeholder(tf.int64, shape=[None, None, 2], name="targets")
        targets = self.targets

        conv_layer = [[5, 1, 32, 2, 2],[5, 1, 64, 2, 2]]

        deep_conv = DeepConv(conv_layer).deep_conv_layers(1, inputs)
        
        with tf.variable_scope("rnn_layer"):
            num_feature = int(img_height / 4.0 * 64) # two conv layers, pooling twice: 2 * 2 = 4, seconde out_channel is 64
            seq_len = seq_len / 4
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_1, state_is_tuple=True) 
            output_1, _ = tf.nn.dynamic_rnn(rnn_cell, tf.reshape(deep_conv, [batch_size, -1, num_feature]), 
                    seq_len, dtype=tf.float32, parallel_iterations=batch_size)

        output_1 = tf.reshape(output_1, [-1, num_hidden_1])
        with tf.variable_scope("logits"):
            w = tf.get_variable("weight",
                    initializer=tf.truncated_normal([num_hidden_1, 2], stddev=0.01))
            b = tf.get_variable("bias",
                    initializer=tf.constant(0.1, shape=[2]))

            outputs = tf.matmul(output_1, w) + b
            outputs = tf.reshape(outputs, [batch_size, -1, 2])
        with tf.variable_scope("label_pool"):
            target_pool = tf.cast(tf.reshape(targets, [batch_size, -1, 2, 1]), tf.float32)
            pool_height = 4
            pool_width = 1
            pool = tf.nn.max_pool(target_pool, ksize=[1, pool_height, pool_width, 1],
                    strides=[1, pool_height, pool_width, 1], padding="SAME")
            targets = tf.cast(tf.reshape(pool, [batch_size, -1, 2]), tf.int64)
        if not is_test:
            # loss
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        
        softmax_outputs = tf.nn.softmax(outputs, dim=-1)
        self.arg_outputs = softmax_outputs
        self.arg_targets = targets
        correct_prediction = tf.equal(tf.argmax(softmax_outputs, 2), tf.argmax(targets, 2))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


class SegNet_crop(object):
    def __init__(self, batch_size, is_test=False):
        self.inputs = tf.placeholder(tf.float32, shape=[None, 32, None, 1], name="inputs")
        inputs = self.inputs
        if not is_test:
            self.targets = tf.placeholder(tf.int64, shape=[None, 2], name="targets")
        else:
            self.targets = tf.placeholder(tf.int64, shape=[None, None], name="targets")

        targets = self.targets

        conv_layer = [[5, 1, 32, 2, 2],[5, 1, 64, 2, 2]]

        deep_conv = DeepConv(conv_layer).deep_conv_layers(1, inputs)

        if not is_test:
            with tf.variable_scope("full"):
                w_f = tf.get_variable('weights',
                        initializer=tf.truncated_normal([8*8*64, 128], stddev=0.01))
                b_f = tf.get_variable('biases',
                        initializer=tf.truncated_normal([128], stddev=0.01))

                deep_conv_flat = tf.reshape(deep_conv, [-1, 8*8*64])
                h_f = tf.nn.relu(tf.matmul(deep_conv_flat, w_f) + b_f)

            with tf.variable_scope("readout"):
                w_r = tf.get_variable('weights',
                        initializer=tf.truncated_normal([128, 2], stddev=0.01))
                b_r = tf.get_variable('biases',
                        initializer=tf.truncated_normal([2], stddev=0.01))

                y_conv = tf.matmul(h_f, w_r) + b_r
        else:
            with tf.variable_scope("full"):
                w_f_test = tf.reshape(tf.get_variable('weights'), [8, 8, 64, 128])
                b_f_test = tf.get_variable('biases')

                h_f_test = tf.nn.conv2d(deep_conv, w_f_test,
                        strides=[1, 8, 1, 1], padding="SAME")
                h_conv = tf.nn.relu(tf.nn.bias_add(h_f_test, b_f_test))

            with tf.variable_scope("readout"):
                w_r_test = tf.reshape(tf.get_variable('weights'), [1, 1, 128, 2])
                b_r_test = tf.get_variable('biases')

                h_r_test = tf.nn.conv2d(h_conv, w_r_test, 
                        strides=[1, 1, 1, 1], padding='SAME')
                y_conv = tf.nn.relu(tf.nn.bias_add(h_r_test, b_r_test))
                y_conv = tf.reshape(y_conv, [batch_size, -1, 2])

            with tf.variable_scope("label_pool"):
                target_pool = tf.cast(tf.reshape(targets, [batch_size, -1, 1, 1]), tf.float32)
                pool_height = 4
                pool_width = 1
                pool = tf.nn.max_pool(target_pool, ksize=[1, pool_height, pool_width, 1],
                        strides=[1, pool_height, pool_width, 1], padding="SAME")
                pool = tf.cast(pool, dtype=tf.int32)

                '''
                targets = tf.one_hot(tf.reshape(pool, [batch_size, -1]), depth=2, 
                        on_value=1, off_value=0, axis=-1, dtype=tf.int32)
                '''
                targets = tf.cast(pool, dtype=tf.int64)

        if not is_test:
            # loss
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, targets))
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        
            y_softmax = tf.nn.softmax(y_conv, dim=-1)
            self.argmax_outputs = tf.argmax(y_softmax, 1)
            self.argmax_targets = tf.argmax(targets, 1)

        else:
            y_softmax = tf.nn.softmax(y_conv, dim=-1)
            self.argmax_outputs = tf.argmax(y_softmax, 2)
            self.argmax_targets = targets

        correct_prediction = tf.equal(self.argmax_outputs, self.argmax_targets)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def RecNet(object):
    def __init__(self, is_test=False):
        pass    


