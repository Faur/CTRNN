from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import _linear 
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py

class CTRNNCell(tf.nn.rnn_cell.RNNCell):
    """ API Conventions: https://giathub.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py
    """
    # TODO: Make a method that creates a zero state_tuple for initialization
    def __init__(self, num_units, tau, activation=None):
        self._num_units = num_units
        self.tau = tau
#         TODO: Implement: Possible values of tau:
#             * constant scalar
#             * variable scalar (learnable, shared for entire layer)
#             * variable vector (learnable, individually)
        if activation is None:
            self.activation = lambda x: 1.7159 * tf.tanh(2/3*x)
            # from: LeCun et al. 2012: Efficient backprop
        else:
            self.activation = activation


    @property # Function is callable without (), as if it was a property...
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype=tf.float32):
        """ Returns a zero filled tuple with shapes equivalent to (new_c, new_u)"""
        zero_c = tf.zeros([batch_size, self.state_size], dtype=dtype)
        zero_u = tf.zeros([batch_size, self.state_size], dtype=dtype)
        return (zero_c, zero_u)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):

            old_c = state[0]
            old_u = state[1]
#             inputs = tf.expand_dims(inputs, axis=1)
#             concat_vector = tf.concat([inputs, old_c], axis=1)
#             print('concat_vector', concat_vector.get_shape())
#             print('inputs', inputs.get_shape())
#             print('old_c', old_c.get_shape())
#             with tf.variable_scope('linear'):
#                 W = tf.get_variable('W', [num_units, output_dim])
#                 b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
#                 logits = tf.matmul(concat_vector, W) + b
            with tf.variable_scope('linear'):
                logits = _linear([inputs, old_c], output_size=self.output_size, bias=True)

            with tf.variable_scope('applyTau'):
                new_u = (1-1/self.tau)*old_u + 1/self.tau*logits

            new_c = self.activation(new_u)

        return new_c, (new_c, new_u)


class CTRNNModel(object):
    def __init__(self, num_steps, input_dim, num_units, output_dim, tau=1, learning_rate=1e-4):
        """ Assumptions
            * x is 3 dimensional: [batch_size, num_steps] 

        """
        self.tau = tau
        self.num_units = num_units 
        self.output_dim = output_dim 
        self.activation = lambda x: 1.7159 * tf.tanh(2/3 * x)

        self.x = tf.placeholder(tf.float32, shape=[None, num_steps, input_dim], name='inputPlaceholder')
        self.y = tf.placeholder(tf.int32, shape=[None, num_steps], name='outputPlaceholder')
        self.y_reshaped = tf.reshape(tf.transpose(self.y, [1,0]), [-1])
        init_c1 = tf.placeholder(tf.float32, shape=[None, num_units], name='initC1')
        init_c2 = tf.placeholder(tf.float32, shape=[None, num_units], name='initC2')
        init_u = tf.placeholder(tf.float32, shape=[None, num_units], name='initU')
        self.init_tuple = (init_c1, (init_c2, init_u))
        # x = tf.one_hot(x, input_dim) # TODO: This should probably not be hard coded

        self.cell = CTRNNCell(num_units, tau=self.tau, activation=self.activation)

        # print('x', self.x.get_shape())
        # print('init_tuple', type(self.init_tuple))
        # print('init_tuple[0]', self.init_tuple[0].get_shape())
        # print('init_tuple[1][0]', self.init_tuple[1][0].get_shape())
        # print('init_tuple[1][1]', self.init_tuple[1][1].get_shape())
        
        self.rnn_outputs, self.final_states = tf.scan(
            lambda state, x: self.cell(x, state[1]),
            tf.transpose(self.x, [1, 0, 2]),
            # tf.transpose(x, [1, 0] + [i+2 for i in range(x_shape.shape[0]-2)]),
                # We need shape = [num_seq, batch_size, ...]
            initializer=self.init_tuple
        )

#         print('self.rnn_outputs[-1]', self.rnn_outputs[-1].shape)
#         print('self.final_states', type(self.final_states))
#         print('self.final_states[0][-1]', self.final_states[0][-1].shape)
#         print('self.final_states[1][-1]', self.final_states[1][-1].shape)
        self.state_tuple = (self.rnn_outputs[-1], 
                           (self.final_states[0][-1], self.final_states[1][-1]))

        rnn_outputs = tf.reshape(self.rnn_outputs, [-1, num_units])

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [num_units, output_dim])
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
            self.logits = tf.matmul(rnn_outputs, W) + b
            self.softmax = tf.nn.softmax(self.logits, dim=-1)

        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.y_reshaped))

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)

    def zero_state_tuple(self, batch_size):
        """ Returns a tuple og zeros with shapes:
                [rnn_output.shape, final_state.output]
        """
        output = np.zeros([batch_size, self.num_units])
        state = np.zeros([batch_size, self.num_units])
        return (output, (output, state))





