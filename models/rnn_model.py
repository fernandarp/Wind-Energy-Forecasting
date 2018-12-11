import pandas as pd
import numpy as np
import tensorflow as tf

class RNNModel():
    def __init__(self, data, test_size):
        self.test_size = test_size
        self.y_data_complete = data
        self.train = data[:len(data)-test_size]
        self.test = data[len(data)-test_size:]
    
    def _get_train_batch(self, num_periods = 20, f_horizon = 1, num_inputs = 1):
        self.input = self.train[:(len(self.train) - (len(self.train) % num_periods))] #making possible to reshape the input based on the num_periods given
        x_batches = self.input.reshape(-1, num_periods, num_inputs)

        self.output = self.train[1:(len(self.train) - (len(self.train) % num_periods)) + f_horizon] #shifting the input to create the output
        y_batches = self.output.reshape(-1, num_periods, 1)

        return x_batches, y_batches
    
    def run_RNN(self, max_time, num_neurons, inputs, output, learning_rate, epochs):
        x_batches , y_batches = self._get_train_batch(max_time, 1, inputs)
        
        tf.reset_default_graph()
        
        X = tf.placeholder(tf.float32, [None, max_time, 1])
        y = tf.placeholder(tf.float32, [None, max_time, 1])
        
        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = num_neurons, activation = tf.nn.relu)
        rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

        stacked_rnn_output = tf.reshape(rnn_output, [-1, num_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
        outputs = tf.reshape(stacked_outputs, [-1, max_time, output])

        loss = tf.reduce_mean(tf.square(outputs - y))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            init.run()
            for ep in range(epochs):
                sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
                if ep % 100 == 0:
                    mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
                    print(ep, "\tMSE: ", mse)

            outputs_array = []
            test = self.y_data_complete[-(max_time + 1):][:max_time].reshape(-1, max_time, inputs)
            
            for i in range(0, len(self.y_data_complete) - len(self.train)):
                y_pred = sess.run(outputs, feed_dict = {X: test})
                y_pred_last = y_pred[:, -1, :]

                outputs_array.append(y_pred_last[0][0])

                test = np.append(test[:,1:max_time,:], [[[y_pred_last]]]).reshape(-1, max_time, inputs)
        
        return np.array(outputs_array)