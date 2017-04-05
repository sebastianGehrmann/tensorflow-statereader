"""
This is a modified version of https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb

author: Sebastian Gehrmann

This file trains a simple language model and extracts the states from the training set.
The extracted states can be used for LSTMVis (https://github.com/HendrikStrobelt/LSTMVis)

For a description how to use the code, please look at
https://github.com/sebastianGehrmann/tensorflow-statereader

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import h5py

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq, framework, legacy_seq2seq

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string("load_path", None,
                    "Checkpoint directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

# Hyperparameters
flags.DEFINE_float("init_scale", 0.1, "TBD")
flags.DEFINE_float("learning_rate", 1.0, "initial learning rate")
flags.DEFINE_integer("max_grad_norm", 5, "Max norm of the gradient")
flags.DEFINE_integer("num_layers", 2, "Layers of the LSTM")
flags.DEFINE_integer("num_steps", 30, "Steps to unroll the LSTM")
flags.DEFINE_integer("hidden_size", 200, "Cell states")
flags.DEFINE_integer("max_epoch", 4, "How many epochs with max LR")
flags.DEFINE_integer("max_max_epoch", 10, "How long to train for")
flags.DEFINE_float("dropout", 1.0, "Dropout")
flags.DEFINE_float("lr_decay", 0.5, "Multiplier of LR")
flags.DEFINE_integer("batch_size", 20, "Batchsize")
flags.DEFINE_integer("vocab_size", 6500, "Size of Vocabulary")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """The input data."""

    def __init__(self, data, name=None):
        self.batch_size = batch_size = FLAGS.batch_size
        self.num_steps = num_steps = FLAGS.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        self._size = FLAGS.hidden_size
        vocab_size = FLAGS.vocab_size

        lstm_cell = rnn.BasicLSTMCell(
            self._size, forget_bias=0.0, state_is_tuple=True)
        if is_training and FLAGS.dropout < 1:
            lstm_cell = rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=FLAGS.dropout)
        cell = rnn.MultiRNNCell(
            [lstm_cell] * FLAGS.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, self._size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and FLAGS.dropout < 1:
            inputs = tf.nn.dropout(inputs, FLAGS.dropout)

        outputs = []
        self._states = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
                self._states.append(state)

        output = tf.reshape(tf.concat(outputs, 1), [-1, self._size])
        softmax_w = tf.get_variable(
            "softmax_w", [self._size, vocab_size], dtype=data_type())
        # print(softmax_w, "SOFTMAX W")
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        # print(softmax_b, "SOFTMAX B")
        logits = tf.matmul(output, softmax_w) + softmax_b  # 400 x 10k
        # print(logits, "LOGITS")
        # print(input_.targets, "TARGET")
        loss = legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          FLAGS.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def states(self):
        return self._states

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def size(self):
        return self._size


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,

    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    all_states = []
    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals, stat = session.run([fetches, model.states], feed_dict)
        curr_states = np.array([s[0][0] for s in stat])
        if len(all_states) == 0:
            all_states = curr_states
        else:
            all_states = np.vstack((all_states, curr_states))
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters), all_states


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = reader.ptb_raw_data(FLAGS.data_path, True)
    train_data, valid_data, _ = raw_data

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
                                                    FLAGS.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Train_states"):
            train_input = PTBInput(data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mstates = PTBModel(is_training=False, input_=train_input)
            tf.summary.scalar("Training Loss", mstates.cost)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            if FLAGS.load_path:
                sv.saver.restore(session, tf.train.latest_checkpoint(FLAGS.load_path))
            else:
                for i in range(FLAGS.max_max_epoch):
                    lr_decay = FLAGS.lr_decay ** float(max(i + 1 - FLAGS.max_epoch, 0))
                    m.assign_lr(session, FLAGS.learning_rate * lr_decay)

                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                    train_perplexity, stat = run_epoch(session, m, eval_op=m.train_op,
                                                       verbose=True)
                    print(stat.shape)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                    valid_perplexity, stat = run_epoch(session, mvalid)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            # run and store the states on training set
            train_perplexity, stat = run_epoch(session, mstates, eval_op=m.train_op,
                                               verbose=True)
            f = h5py.File("states.h5", "w")
            stat = np.reshape(stat, (-1, mstates.size))
            f["states1"] = stat
            f.close()

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
