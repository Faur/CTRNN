from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import _linear 
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py

class CTRNNCell(tf.nn.rnn_cell.RNNCell):
    """ API Conventions: https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py
    """
    def __init__(self, num_units, tau, activation=None):
        self._num_units = num_units
        self.tau = tau
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

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.

        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
        the shapes `[batch_size x s]` for each s in `state_size`.
        """
        state_size = self.state_size
        if nest.is_sequence(state_size):
            state_size_flat = nest.flatten(state_size)
            zeros_flat = [
                array_ops.zeros(
                    array_ops.stack(_state_size_with_prefix(s, prefix=[batch_size])),
                    dtype=dtype)
                for s in state_size_flat]
            for s, z in zip(state_size_flat, zeros_flat):
                z.set_shape(_state_size_with_prefix(s, prefix=[None]))
            zeros = nest.pack_sequence_as(structure=state_size,
                                        flat_sequence=zeros_flat)
        else:
            zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
            zeros = array_ops.zeros(array_ops.stack(zeros_size), dtype=dtype)
            zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

        return zeros


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            old_c = state[0]
            old_u = state[1]
            # print(scope)
            # print('inputs', len(inputs), inputs[0].get_shape())
            # print('state', type(state))
            # print('state[0]', state[0].get_shape())
            # print('state[1]', state[1].get_shape())
            # print()

            with tf.variable_scope('linear'):
                logits = _linear(inputs + [old_c], output_size=self.output_size, bias=True)

            with tf.variable_scope('applyTau'):
                new_u = (1-1/self.tau)*old_u + 1/self.tau*logits

            new_c = self.activation(new_u)

        return new_c, (new_c, new_u)

def shape_printer(obj, prefix):
    try:
        print(prefix, obj.shape)
    except AttributeError:
        print(prefix, type(obj))
        for o in obj:
            shape_printer(o, prefix + '\t')


class MultiLayerHandler():
    def __init__(self, layers):
        """ layers: A list of layers """
        self.layers = layers
        self.num_layers = len(layers)

    @property # Function is callable without (), as if it was a property...
    def state_size(self):
        raise NotImplementedError
        # num_units = []
        # for l in self.layers:
        #     num_units += l.state_size
        # return num_units

    @property
    def output_size(self):
        raise NotImplementedError
        # return self.layers[0]._num_units

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.

        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
        the shapes `[batch_size x s]` for each s in `state_size`.
        """
        raise NotImplementedError
        # """ Returns a zero filled tuple with shapes equivalent to (new_c, new_u)"""
        # zero_states = []
        # for l in self.layers:
        #     zero_states += l.zero_state(batch_size)
        # return zero_states

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or type(self).__name__):
            out_state = []
            for i_, l in enumerate(reversed(self.layers)): # Start with the top level
                i = self.num_layers - i_ - 1
                scope = 'CTRNNCell_' + str(i)

                cur_state = state[i]
                if i == 0: # IO level, last executed
                    print('IO level')
                    cur_input = [inputs] + [state[i+1][0]]
                elif i == self.num_layers - 1: # Highest level
                    print('Highest level')
                    cur_input = [state[i-1][0]]
                    # print(cur_input)
                else: # Inbetween layers
                    cur_input = [state[i-1][0]] + [state[i+1][0]]

                outputs, state_ = l(cur_input, cur_state, scope=scope)
                # print('state_', type(state_))
                # print('state_[0]', state_[0].get_shape())
                out_state += [state_]

            out_state = tuple(out_state)

            # shape_printer(out_state, 'MLH')

            # print('outputs', outputs.get_shape())
            # print('out_state[0][0]', out_state[0][0].get_shape())
            return outputs, out_state

        # with tf.variable_scope(scope or type(self).__name__):
        #     for i, l in enumerate(self.layers):
        #         scope = 'CTRNNCell_' + str(i)
        #         inputs, state = l([inputs], state, scope=scope)
        # return inputs, state


class CTRNNModel(object):
    def __init__(self, num_units, tau, num_steps, input_dim, output_dim, learning_rate=1e-4):
        """ Assumptions
            * x is 3 dimensional: [batch_size, num_steps] 

            Args:
            * num_units: list with num_units, with num_units[0] being the IO layer
            * taus: list with tau values (also if it is only one element!)
        """
        self.num_units = num_units
        self.num_layers = len(self.num_units)
        self.tau = tau

        self.output_dim = output_dim 
        self.activation = lambda x: 1.7159 * tf.tanh(2/3 * x)

        self.x = tf.placeholder(tf.float32, shape=[None, num_steps, input_dim], name='inputPlaceholder')
        self.y = tf.placeholder(tf.int32, shape=[None, num_steps], name='outputPlaceholder')
        self.y_reshaped = tf.reshape(tf.transpose(self.y, [1,0]), [-1])


        init_input = tf.placeholder(tf.float32, shape=[None, self.num_units[0]], name='initInput')
        init_state = []
        for i, num_unit in enumerate(self.num_units):
            init_c = tf.placeholder(tf.float32, shape=[None, num_unit], name='initC_' + str(i))
            init_u = tf.placeholder(tf.float32, shape=[None, num_unit], name='initU_' + str(i))
            init_state += [(init_c, init_u)]
        init_state = tuple(init_state)
        # print('init_input', init_input.get_shape())
        # print('init_state[0][0]', init_state[0][0].get_shape())
        # print()
        self.init_tuple = (init_input, init_state)
        # self.init_tuple = (init_input, init_state[0])

        # init_c = tf.placeholder(tf.float32, shape=[None, num_units[0]], name='initC_')
        # init_u = tf.placeholder(tf.float32, shape=[None, num_units[0]], name='initU_')
        # self.init_tuple = (init_input, (init_c, init_u))
        # print(init_state[0])
        # print((init_c, init_u))

        cells = []
        for i in range(self.num_layers): 
            num_unit = num_units[i]
            tau = self.tau[i]
            cells += [CTRNNCell(num_unit, tau=tau, activation=self.activation)]
        self.cell = MultiLayerHandler(cells) # First cell (index 0) is IO layer

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

        # print('self.rnn_outputs[-1]', self.rnn_outputs[-1].shape)
        # print('self.final_states', type(self.final_states))
        # print('self.final_states[0][-1]', self.final_states[0][-1].shape)
        # print('self.final_states[1][-1]', self.final_states[1][-1].shape)

        # print('shape_printer: self.final_states')
        # shape_printer(self.final_states, 'fs')

        # self.state_tuple = (self.rnn_outputs[-1], 
        #                    (self.final_states[0][-1][-1], self.final_states[1][-1][-1]))

        state_state = []
        for i in range(self.num_layers):
            state_state += [(self.final_states[i][0][-1], self.final_states[i][1][-1])]
        state_state = tuple(state_state)
        self.state_tuple = (self.rnn_outputs[-1], state_state)

        # print('shape_printer: self.state_tuple')
        # shape_printer(self.state_tuple, 'st')


        rnn_outputs = tf.reshape(self.rnn_outputs, [-1, num_units[0]])

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [num_units[0], output_dim])
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
            self.logits = tf.matmul(rnn_outputs, W) + b
            self.softmax = tf.nn.softmax(self.logits, dim=-1)

        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.y_reshaped))
        tf.summary.scalar('training/total_loss', self.total_loss)

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)
        self.TBsummaries = tf.summary.merge_all()

    def zero_state_tuple(self, batch_size):
        """ Returns a tuple og zeros
        """
        zero_input = np.zeros([batch_size, self.num_units[0]])

        zero_state = []
        for i, num_unit in enumerate(self.num_units):
            zero_c = np.zeros([batch_size, self.num_units[i]])
            zero_u = np.zeros([batch_size, self.num_units[i]])
            zero_state += [(zero_c, zero_u)]

        zero_state = tuple(zero_state)
        return (zero_input, zero_state)
        # output = np.zeros([batch_size, self.num_units[0]])
        # state = np.zeros([batch_size, self.num_units[0]])
        # return (output, (output, state))





