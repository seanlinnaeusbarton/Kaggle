# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:26:30 2018

@author: Travel


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import datetime as dt

""" NOTES
'batch' = A collection of example word sequences (x) and the NEXT sequence
    that is to be predicted (y). If I have a DATA_SET of 10 words, and my
    BATCH_SIZE is 2, the maximum length of any patricular batch is given by
    the integer division of the length of my DATA_SET and my BATCH_SIZE (i.e.,
    len(DATA_SET) // BATCH_SIZE).
    
'epoch' = The number of times an *entire* DATA_SET is passed forward and
    backward (back-prop) through a learning routine. The entire DATA_SET must
    be passed through multiple times for effective learning (i.e., an
    algorithm must see each [x,y] pair more than once).
    
'batched length' = The length of the data after it has been folded into
    'batches'. 
"""


### Utilties for Reading Data (PTB) ###
Py3 = sys.version_info[0] == 3

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()

def build_vocab(filename):
    data = read_words(filename)
    
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    
    return word_to_id

def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_data(data_path="./simple-examples/data"):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    
    print(train_data[:5])
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

def batch_producer(raw_data, batch_size, num_steps):
    raw_data_tensor = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    
    data_len = tf.size(raw_data_tensor)
    batched_len = data_len // batch_size #formerly 'batch_len'
    data = tf.reshape(raw_data_tensor[0:batch_size * batched_len], 
                      [batch_size, batched_len])
    
    epoch_size = (batched_len - 1) // num_steps
    
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1)*num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i*num_steps + 1:(i+1)*num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y

class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)

class Model(object):
    def __init__(self, input_obj, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input_obj
        self.batch_size = input_obj.batch_size
        self.num_steps = input_obj.num_steps
        self.hidden_size = hidden_size # size of word embedding
        self.vocab_size = vocab_size
        
        ## Create the word embeddings
        #       The initial word embedding vector (which could also be pre-
        #       loaded) is given by a uniformally random tensor of size
        #       [vocab, embedding_dim].
        with tf.device("/cpu:0"):
            # Initialize embedding tensor (to be learned in this case)
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size],
                                                      -init_scale, init_scale))
            # Feed input data into embedding to create look-up table
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)
        
        # Dropout wrapper to prevent over-fitting
        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)
            
        ## Storage variable for LSTM "state" information
        self.init_state = tf.placeholder(tf.float32,
                                         [num_layers, 2, None, self.hidden_size])#self.batch_size, self.hidden_size])
        
        # Convert state tensor into a list of state tensors (one for each layer)
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple([
                rnn.LSTMStateTuple(
                        state_per_layer_list[layer][0], # 's' (sometimes 'c'?) <- pervious state
                        state_per_layer_list[layer][1] # 'h' <- previous output
                ) for layer in range(num_layers)
        ])
                
        ## Create LSTM cells
        cell = rnn.LSTMCell(hidden_size, forget_bias=1.0)
        # Add drop-out wrapper if training
        if is_training and dropout < 1:
            cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        
        # Create a list of LSTM cells representing each layer
        if num_layers > 1:
            cells = [cell for _ in range(num_layers)]
            cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
            
        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        output = tf.reshape(output, [-1, hidden_size]) # Reshape to [batch_size*num_steps , hidden_size]
        
        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale)) # Initial softmax weights
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale)) # Initial softmax biases
        
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b) # output * weights + biases
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.vocab_size])
        
        ## Calculate Loss
        loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                self.input_obj.targets,
                tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
                average_across_timesteps=False,
                average_across_batch=True)
        
        self.cost = tf.reduce_sum(loss)
        
        ## Get prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32) # Transform softmax into integer
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1])) # Check if prediction was correct
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        if not is_training:
            return
        
        self.learning_rate = tf.Variable(0.0, trainable=False)
        
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        
        self.train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())
        
        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)
        
    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})
        
def train(train_data, vocabulary, num_layers, num_epochs, batch_size, model_save_name,
          learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=50, 
          hidden_size=650, num_steps=35, model_path="null"):
    
    tf.reset_default_graph()
    ## Setup data and models
    training_input = Input(batch_size=batch_size, num_steps=num_steps, data=train_data)
    m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocabulary,
              num_layers=num_layers)
    
    init_op = tf.global_variables_initializer()
    orig_decay = lr_decay
    
    with tf.Session() as sess:
        ## Start threads
        sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        
        if model_path != 'null':
            saver.restore(sess,model_path)
        
        for epoch in range(num_epochs):
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(sess, learning_rate * new_lr_decay)
            
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
            curr_time = dt.datetime.now()
            
            for step in range(training_input.epoch_size):
                if step % print_iter != 0:
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                           feed_dict={m.init_state: current_state})
                else:
                    seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = dt.datetime.now()
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                           feed_dict={m.init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(epoch,
                          step, cost, acc, seconds))
            # Save model checkpoint
            saver.save(sess, data_path + '\\' + model_save_name, global_step=epoch)
        #do a final save
        saver.save(sess, data_path + '\\' + model_save_name + '-final')
        # close threads
        coord.request_stop()
        coord.join(threads)
            
def test(model_path, test_data, reversed_dictionary):
    tf.reset_default_graph()
    test_input = Input(batch_size=20, num_steps=35, data=test_data)
    m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocabulary,
              num_layers=2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
        # restore the trained model
        saver.restore(sess, model_path)
        # get an average accuracy over num_acc_batches
        num_acc_batches = 30
        check_batch_idx = 25
        acc_check_thresh = 5
        accuracy = 0
        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true_vals, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy],
                                                               feed_dict={m.init_state: current_state})
                pred_string = [reversed_dictionary[x] for x in pred[:m.num_steps]]
                true_vals_string = [reversed_dictionary[x] for x in true_vals[0]]
                print("True values (1st line) vs predicted values (2nd line):")
                print("###########")
                print(" ".join(true_vals_string))
                print("###########")
                print(" ".join(pred_string))
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
            if batch >= acc_check_thresh:
                accuracy += acc
        print("Average accuracy: {:.3f}".format(accuracy / (num_acc_batches-acc_check_thresh)))
        # close threads
        coord.request_stop()
        coord.join(threads)


#if args.data_path:
#    data_path = args.data_path
#train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
#if args.run_opt == 1:
#    train(train_data, vocabulary, num_layers=2, num_epochs=60, batch_size=20,
#          model_save_name='two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr')
#else:
#    trained_model = args.data_path + "\\two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr-38"
#test(trained_model, test_data, reversed_dictionary)     
        
        

if __name__ == "__main__":
    data_path = "C:\\Users\\Travel\\Desktop\\Sean\\ALLIES\\ALLIES_Repo\\Experimental\\NN_CCM\\Text_Prediction"
    trained_model = data_path + "\\two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr-7"
    
    train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
    
    train(train_data, vocabulary, num_layers=3, num_epochs=60, batch_size=20,
          model_save_name='two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr',
          hidden_size=1300)
    
#    test(trained_model, test_data, reversed_dictionary)    





