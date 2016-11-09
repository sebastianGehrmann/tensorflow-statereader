#!/usr/bin/env python

"""
Create and train a recurrent model with tensorflow
"""

import argparse
import h5py
import numpy as np
import sys
import tensorflow as tf

from tensorflow.python.ops import rnn, rnn_cell


__author__ = 'Sebastian Gehrmann'
global args


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    data_opt = parser.add_argument_group("Data Options")
    data_opt.add_argument('--data', help="The h5 File with Training Data", default="train.h5", type=str)
    data_opt.add_argument('--val_data', help="The h5 File with Validation Data", default="val.h5", type=str)
    data_opt.add_argument('--savefile', help="Filename of Saved Trained Model", default="val.h5", type=str)
    model_opt = parser.add_argument_group("Model Options")
    model_opt.add_argument('--rnn_size', help="Hidden Layers", default=100, type=int)
    model_opt.add_argument('--word_vec_size', help="Word Embedding Size", default=50, type=int)
    model_opt.add_argument('--layers', help="Number of Layers", default=2, type=int)
    model_opt.add_argument('--type', help="Type of Model: GRU/LSTM", default="LSTM", type=str)
    train_opt = parser.add_argument_group("Training Options")
    train_opt.add_argument('--epochs', help="Number of Epochs", default=10, type=int)
    train_opt.add_argument('--learning_rate', help="Initial Learning Rate", default=0.01, type=float)
    train_opt.add_argument('--max_grad_norm', help="Gradient L2-Normalization", default=5, type=int)
    train_opt.add_argument('--dropout', help="Dropout Probability", default=0.5, type=float)
    train_opt.add_argument('--param_init', help="Parameter Initialization", default=0.05, type=float)
    args = parser.parse_args(arguments)


    train = h5py.File(args.data, 'r')
    val = h5py.File(args.val_data, 'r')

    V = train['target_size']
    print V

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

# cmd:option('-savefile', 'lm_word','Filename to autosave the checkpont to')
#
# opt = cmd:parse(arg)