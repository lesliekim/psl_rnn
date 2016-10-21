import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc  # ctc no more in contrib
from tensorflow.contrib import grid_rnn


class CnnRnnModel(object):
    def __init__(self, batch_size=1):
        # Network configs
        momentum = 0.9
        learning_rate = 1e-4
        model_type = 'gru'
        image_height = 32 #48
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
        image_height = 32 #48
        # Accounting the 0th indice + blank label = 121 characters
        num_classes = 128 #121
        num_hidden_1 = 50
        num_hidden_2 = 100
        num_hidden_3 = 200

        # Has size [batch_size, max_stepsize, num_features], but the
        # batch_size and max_stepsize can vary along each step
        self.inputs = tf.placeholder(tf.float32, [None, None, image_height, 1], name="inputs")
        inputs = tf.reshape(self.inputs, [tf.shape(self.inputs)[0], -1, image_height])
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

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

        loss = ctc.ctc_loss(logits, self.targets, tf.cast(self.seq_len, dtype='int32'))
        self.cost = tf.reduce_mean(loss)

        self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.cost)


        self.decoded, log_prob = ctc.ctc_beam_search_decoder(logits, tf.cast(self.seq_len, dtype='int32'))


        # Accuracy: label error rate
        self.err = tf.reduce_sum(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets, normalize=False))

        self.saver = tf.train.Saver(max_to_keep=0)
