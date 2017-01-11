# tensorflow-statereader

This Repository provides a simple LSTM implementation including a state extractor. A model is first trained and states then extracted and stored in a hdf5 file. This makes it possible to train custom language models for [LSTMVis](https://github.com/HendrikStrobelt/LSTMVis). 

This code is heavily based on tutorial implementation in the official documentation which can be found [here](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb).

A standard model using the Penn Treebank can be trained by simply running ``python lstm.py --data/ptb-word``. In case you want to train the model with custom parameters, your own data or load your own model, we provide the following options.

## Parameters

```
  --data_path       The folder where your training/validation data is stored.
  --save_path       The code saves the trained model to this directory.
  --load_path       If you want to load a trained model, enter its folder here.
  --use_fp16        Train using 16-bit floats instead of 32bit floats. [False]
 
  --init_scale      Scale of the uniform parameter initialization. [0.1]
  --learning_rate   The initial learning rate. [1.0]
  --max_grad_norm   Max norm of the gradient. [5]
  --num_layers      Layers of the LSTM. [2]
  --num_steps       Steps to unroll the LSTM for. [30]
  --hidden_size     Number of cell states. [200]
  --max_epoch       How many epochs with max learning rate before decay begins. [4]
  --max_max_epoch   How many epochs to train for. [10]
  --dropout         Dropout probability. [1.0]
  --lr_decay        Decay multiplier for the learning rate. [0.5]
  --batch_size      Batchsize. [20]
  --vocab_size      Size of Vocabulary [6500]   
```
The standard parameters lead to a very small model that is quickly trained. For parameter configuration for a large model, have a look at http://arxiv.org/abs/1409.2329 by Zaremba et al. 

## Credits

LSTMVis and all its parts are a collaborative project of Hendrik Strobelt, Sebastian Gehrmann, Hanspeter Pfister, and Alexander M. Rush at Harvard SEAS.
