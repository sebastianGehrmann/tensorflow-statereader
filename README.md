# tensorflow-statereader

This Repository provides a simple LSTM implementation including a state extractor for tensorflow 1.1.0. A model is first trained and states then extracted and stored in a hdf5 file. This makes it possible to train custom language models for [LSTMVis](https://github.com/HendrikStrobelt/LSTMVis). 

This code is heavily based on tutorial implementation in the official documentation which can be found [here](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb).

A standard model using the Penn Treebank can be trained by simply running ``python lstm.py --data/ptb-word``. In case you want to train the model with custom parameters, or your own data, we provide the following options.

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
The standard parameters lead to a very small model that is quickly trained. For parameter configuration for a large models, have a look at http://arxiv.org/abs/1409.2329 by Zaremba et al. 

## How to Extract States from Your Model? 

You might be interested in analyzing your own models in LSTMVis. This is easily possible, as long as you can extract some vector that evolves over time, such as hidden states or cell states in an LSTM (or a GRU). To extract your own states, you actually need to add very little code to your model, all of which is documented here. 

### 1. Make sure that you can access the states

It is recommended to build your model in a class and use the `@property` annotation for a get-function. This makes it possible to access the states when you call `session.run`. In our case, we found it easier to unroll the RNN for *n* timesteps instead of using the rnn cell built into tensorflow (**Note:** This is only recommended for the analysis, for best performance, keep using the rnn cell). 

Before: 

    inputs = tf.unstack(inputs, num=num_steps, axis=1)
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
   
After: 
    
    self._states = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
            self._states.append(state)
    

With the code above, you have a variable that stores an an array of length $num_steps$ that contains LSTMStateTuple, a tensorflow-internal data type that stores *c*, the cell states, and *h*, the hidden states. The states are of dimension *batch_size * state_size*. 

You get this array by executing your existing `session.run` code, with the following addition: 

Old:

    vals = session.run(fetches, feed_dict)
    
New:

    vals, stat = session.run([fetches, model.states], feed_dict)

This gives you the states for one batch, stored in a variable `stat`. 

### 2. Get all states sequentially and store them

To store all the cell states for a data set, use the `numpy.vstack` function to add them all to an array, which we call `all_states`. We also have to transform stat into an array to get rid of the LSTMStateTuple overhead. The first line of the following code takes care of this. If you want to access hidden states instead of the cell states, change the `s[0][0]` to `s[0][1]`. 

    curr_states = np.array([s[0][0] for s in stat])
    if len(all_states) == 0:
        all_states = curr_states
    else:
        all_states = np.vstack((all_states, curr_states))
        
After the whole epoch, `all_states` contains a tensor of the form *num_steps * batch_size * state_size*. To transform them into an array of the dimensions *data_length * state_size*, use `stat = np.reshape(stat, (-1, stat.shape[2]))`. The last step is storing the states in a hdf5 file, which can be achieved with the following code: 

    f = h5py.File("states.h5", "w")
    f["states1"] = stat
    f.close()
    
Congrats, you now have stored your cell states in a LSTMVis compatible format! 





## Credits

LSTMVis and all its parts are a collaborative project of Hendrik Strobelt, Sebastian Gehrmann, Hanspeter Pfister, and Alexander M. Rush at Harvard SEAS.
