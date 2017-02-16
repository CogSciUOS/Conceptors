# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:40:39 2017

@author: rgast
"""

import tensorflow as tf
import numpy as np
import os
import inspect
from matplotlib.pyplot import *

class DeepNN:
    """
    
    :Description:
        Class that allows to build, train and test a deep neural network based on
        tensorflow. Allows to build graph by adding layers of different type,
        train the free parameters with various training settings and test the
        performance on separate testing data. Also includes several plotting options.
        
    """
    
    def __init__(self, input_dim, output_dim):
        """
        
        :Description:
            Initializes instance of DeepNN.
        
        :Input parameters:
            input_dim:           Size of each input dimension (list)
            output_dim:          Size of each output dimension (list)
            
        """
        
        self.Weights = []
        self.Activations = []
        self.data = tf.placeholder(tf.float32, [None,input_dim])
        self.labels = tf.placeholder(tf.float32, [None, output_dim]) if output_dim > 1 else tf.placeholder(tf.int32, [None])
        
    def applyActivation(self, activations, activation_function):
        """
        
        :Description:
            Applies chosen activation function to given activation data.
            
        :Input parameters:
            activations:            Activation data of certain layer (tensor)
            activation_function:    type of activation function (string). Can be
                                        a) 'tanh'
                                        b) 'relu' for rectified linear unit
                                        c) 'softplus'
                                        d) 'sigmoid'
                                        e) 'softmax'
                                        f) 'none'
            
        """
        
        if activation_function == 'tanh':
            
            activations = 1.7159*tf.nn.tanh(activations * 2./3)
            
        elif activation_function == 'relu':
            
            activations = tf.nn.relu(activations)
            
        elif activation_function == 'softplus':
            
            activations = tf.nn.softplus(activations)
            
        elif activation_function == 'sigmoid':
            
            activations = tf.nn.sigmoid(activations)
            
        elif activation_function == 'softmax':
            
            activations = tf.nn.softmax(activations)
            
        elif activation_function == 'none':
            
            activations = activations
            
        else:
            
            raise ValueError('Activation function not supported!')
            
        return activations 
        
    
    def addLayer(self, n, activation_function = 'tanh', include_bias = False, sd = 0.35, dropout = 0, normalization = None, weights = None):
        """
        
        :Description:
            Adds a layer to the network, including a weight tensor and an activation tensor.
            
        :Input parameters:
            activation_function:    type of activation function to be applied to each unit (string)
            include_bias:           if true, a column of ones will be added to the weights (boolean)
            sd:                     standard deviation of the zero-mean gaussian from which the weights will be drawn (float)
            dropout:                the chance with which each weight will be set to zero for a given training step (float)
            normalization:          the type of normalization imposed on the layer activations. Can be
                                        a) 'softmax' for softmax normalization
                                        b) 'Shift' for de-meaning
                                        c) 'ShiftScale' for de-meaning and standard deviation normalization
            weights:                if provided, will be used as weights of layer instead of drawing from gaussian (tensor)
            
        """
        
        """ initialize weights and use them to calculate layer activations """
        
        if weights: # if weights are provided, use those
            
            weights = tf.mul(tf.ones(weights.shape),weights)
            activations = tf.matmul(self.data, weights) if not self.weights else tf.matmul(self.Activations[-1], weights)
        
        elif not self.Weights: # else if first layer 
            
            weights = tf.Variable(tf.random_normal([self.data.get_shape()[1].value, n], stddev = sd))      
            weights = tf.concat(1,[weights,tf.ones([weights.get_shape()[0],1])]) if include_bias else weights
            activations = tf.matmul(self.data, weights)
            
        else: # for every other layer
            
            weights = tf.Variable(tf.random_normal([self.Activations[-1].get_shape()[-1].value, n], stddev = sd))
            weights = tf.concat(1,[weights,tf.ones([weights.get_shape()[0],1])]) if include_bias else weights
            activations = tf.matmul(self.Activations[-1], weights)
        
        self.Weights.append(weights)
        self.Activations.append(self.applyActivation(activations, activation_function)) # apply activation function on raw activations
        
        """ add dropout and/or normalization """
        
        if dropout:
            
            self.Activations.append(tf.nn.dropout(self.Activations[-1], dropout))
            
        if normalization == 'softmax': # for softmax normalization
            
            self.Activations.append(tf.nn.softmax(self.Activations[-1]))
            
        elif normalization == 'Shift': # for de-meaning
            
            self.Activations[-1] = tf.subtract(self.Activations[-1],tf.reduce_mean(self.Activations[-1]))
            
        elif normalization == 'ShiftScale': # for de-meaning & and rescaling by variance
            
            mu = tf.reduce_mean(self.Activations[-1])
            diff = tf.subtract(self.Activations[-1],mu)
            self.Activations[-1] = tf.div(diff,tf.reduce_sum(tf.mul(diff,diff)))            
            
        
    def train(self, data, labels, log_dir = None, loss_type = 'MSL', optimizer_type = 'GradientDescent', learning_rate = 0.001, momentum = 0.9,
              n_epochs = 10, batch_size = 50, l1 = 0.0, l2 = 0.0, verbose = 100, check_dir = None, validate_per_step = 1000, 
              plot_perf_step = 0, plot_weights_step = 0, plot_weights_layer = 0, plot_weights_shape = [], plot_weights_bias = False):
        """
        
        :Description:
            Trains the network by going through the training data in random mini-batches for multiple epochs.
            
        :Input parameters:
            data:                       List including two entries, one with training and one with validation data
            labels:                     List including two entries, one with training and one with validation labels corresponding to the data points in data
            log_dir:                    Directory where summaries and checkpoints will be saved and retrieved from (string).
                                        If None, current workding directory will be used.
            loss_type:                  Type of loss that will be used to compute the gradient. Can be
                                            a) 'MSL' for mean squared loss
                                            b) 'L2' for L2 norm loss
                                            c) 'CrossEntropy' for cross entropy loss of labels, where multiple classes can be present for a single data point
                                            d) 'CrossEntropyExclusive' for cross entropy of labels, where only a single class can be present at a time, but other classes can still have probability mass != 0
                                            e) 'CrossEntropyExclusiveSparse' for cross entropy of labels, where labels are expressed as indices for the class each data point belongs to
                                            f) 'Poisson' for data which can be expressed by a poisson distribution
            optimizer_type:             Type of optimizer that will be used to minimize the loss. Can be
                                            a) 'GradientDescent' for standard gradient descent that can be individualized by batch size, learning rate and momentum
                                            b) 'Adadelta' for Adadelta optimizer
                                            c) 'Adagrad' for Adagrad optimizer
                                            d) 'Adam' for Adam optimizer
                                            e) 'ProximalGradientDescent' for gradient descent with l1 and l2 normalization
            learning_rate:              learning rate used for weight updates (float)
            momentum:                   Momentum used for learning rate adaption (float)
            n_epochs:                   Number of times to go through all training data in mini-batches (int)
            batch_size:                 Size of random mini-batches to compute gradient for at each training step (int)
            l1:                         Strength of l1 normalization for proximal gradient descent (float)
            l2:                         Strength of l2 normalization for proximal gradient descent (float)
            verbose:                    Indicates after how many training steps to print the current training loss (int)
            check_dir:                  Sub-directory of log_dir to be used for checkpoint saving (string)
            validate_per_step:          Indicates after how many training steps to evaluate classification performance on validation data (int)
            plot_loss_step:             Indicates after how many training steps to plot classification performance (int)
            plot_weights_step:          Indicates after how many training steps to plot weights of target layer (int)
            plot_weights_layer:         Indicates for which layer to plot weights (int)
            plot_weights_shape:         Indicates in which shape to bring weights before plotting them (list)
            plot_weights_bias:          If true, remove weight for bias before plotting weights (boolean)
            
        """
        
        """ add loss """
        
        if loss_type == 'MSL':
            
            self.loss = tf.reduce_mean(tf.mul(0.5, tf.square(tf.sub(self.Activations[-1], self.labels))))
            
        elif loss_type == 'L2':
            
            self.loss = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(self.Activations[-1], self.labels)))
            
        elif loss_type == 'CrossEntropy':
            
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.Activations[-1], self.labels))
            
        elif loss_type == 'CrossEntropyExclusive':
            
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.Activations[-1], self.labels))
            
        elif loss_type == 'CrossEntropyExclusiveSparse':
            
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.Activations[-1], self.labels))
        
        elif loss_type ==  'Poission':
            
            self.loss = tf.reduce_mean(tf.nn.log_poisson_loss(self.Activations[-1], self.labels))
            
        else:
                
            raise ValueError('Loss type not supported!')
        
        """ add optimizer """
        
        if optimizer_type == 'GradientDescent':
            
            if momentum == 0:
            
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            
            else:
                
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        
        elif optimizer_type == 'Adadelta':
            
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            
        elif optimizer_type == 'Adagrad':
            
            optimizer = tf.train.AdagradOptimizer(learning_rate)
            
        elif optimizer_type == 'Adam':
            
            optimizer = tf.train.AdamOptimizer(learning_rate)
            
        elif optimizer_type == 'ProximalGradientDescent':
            
            optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate, l1, l2)
            
        else:
            
            raise ValueError('Optimizer type not supported!')
        
        """ training procedure """
        
        minimizer = optimizer.minimize(self.loss) # operation that calculates gradient and applies changes to weights
        init = tf.global_variables_initializer() # initialize all variables of network
        
        max_steps = int(len(data[0])/batch_size)
        self.training_error = np.zeros(n_epochs*max_steps)
        self.training_perf = np.zeros_like(self.training_error)
        self.validation_perf = np.zeros_like(self.training_error)
        
        if not log_dir:
            
            log_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            
        self.saver = tf.train.Saver() # method that saves network at certain steps during training
        
        idx = np.arange(0,len(data[0]))
        
        for epoch in range(n_epochs): # run through all training data n_epoch times
            
            np.random.shuffle(idx) # randomize mini-batches to be drawn
                
            print('Epoch #',epoch)
    
            with tf.Session() as sess: # start training session
                
                sess.run(init) if epoch == 0 else self.saver.restore(sess, self.save_dir) # in first epoch, initialize variables freshly, in subsequent epochs, load variables from previous session             
                val_check = 0
                
                for step in range(max_steps): # go trhough training data in mini batches
                    
                    feed_dict = {self.data: data[0][idx[step*batch_size:(step+1)*batch_size]], self.labels: labels[0][idx[step*batch_size:(step+1)*batch_size]]} # extract mini-batch
                    _, self.training_error[epoch*max_steps + step], self.current_weights, prediction = sess.run([minimizer,self.loss,self.Weights, self.Activations[-1]], feed_dict = feed_dict) # run single training step
                    self.training_perf[epoch*max_steps + step] = np.mean(np.argmax(prediction, axis = 1) == labels[0][idx[step*batch_size:(step+1)*batch_size]]) # calculate prediction performance on mini-batch
                    
                    if step % verbose == 0:
                        print('Step %d: loss = %.2f' % (step, self.training_error[epoch*max_steps + step]))
                    
                    if step % validate_per_step == 0 and (step + epoch) != 0: # check performance on validation data for certain training steps
                        
                        feed_dict = {self.data: data[1], self.labels: labels[1]} # extract validation data and labels
                        prediction = sess.run(self.Activations[-1], feed_dict = feed_dict) # run full data through current network
                        self.validation_perf[epoch*max_steps + step-validate_per_step:epoch*max_steps + step] = np.mean(np.argmax(prediction, axis = 1) == labels[1]) # calculate prediction performance on validation data
                    
                        if val_check > 3: # break criterion for decreasing prediction performance on validation data
                            
                            self.saver.restore(sess, self.save_dir) # restore network from checkpoint with best performance on validation data set
                            print('Early stopping criterion reached (validation performance maximum)')
                            break
                        
                        if self.validation_perf[epoch*max_steps+step-1] < self.validation_perf[epoch*max_steps+step-(validate_per_step+1)]: # compare performance on validation data with previous performance
                            
                            val_check += 1
                            
                        else:
                            
                            val_check = 0
                            self.save_dir = self.saver.save(sess, log_dir + '/Checkpoints/' + check_dir +'/log', global_step = epoch*max_steps + step) if check_dir else self.saver.save(sess, log_dir + '/Checkpoints/log', global_step = epoch*max_steps + step) # create checkpoint of current network
                        
                    if plot_perf_step > 0 and step % plot_perf_step == 0: # plot prediction performance on training and validation data
                        
                        self.plot_performance(plt_range = [0,epoch*max_steps+step+plot_perf_step],update = False) if epoch == 0 and step <= plot_perf_step else self.plot_performance(plt_range = [0,epoch*max_steps+step+plot_perf_step],update = True)
                        
                    if plot_weights_step > 0 and step % plot_weights_step == 0: # plot weights of target layer
                        
                        self.plot_weights(plot_weights_layer, plot_weights_shape, update = False, remove_bias = plot_weights_bias) if epoch == 0 and step <= plot_weights_step else self.plot_weights(plot_weights_layer, plot_weights_shape, update = True, remove_bias = plot_weights_bias)
                    
    def test(self, data, labels, normalize = False):
        """
        :Description:
            Function that evaluates classification error and performance on test data.
            
        :Input parameters:
            data:           test data (array)
            labels:         test labels (array)
            normalize:      If true, normalize network activations before computing loss/prediction performance (boolean)
            
        """
        
        if normalize: # apply softmax normalization to activations of output layer 
            
            self.Activations.append(tf.nn.softmax(self.Activations[-1]))
        
        with tf.Session() as sess: # run test session
            
            self.saver.restore(sess, self.save_dir) # restore network from last checkpoint
            
            feed_dict = {self.data: data, self.labels: labels} # extract test data and labels
            self.test_error, self.test_predictions = sess.run([self.loss,self.Activations[-1]], feed_dict = feed_dict) # run test data through network
             
    def plot_loss(self):
        """
        
        :Description:
            Creates plot of training loss.
            
        """
        
        figure()
        plot(self.training_error)
        title('Training Loss')
        xlabel('Training Step')
        ylabel('Loss')

                   
    def plot_performance(self, plt_range = None, update = False):
        """
        
        :Description:
            Creates plot of classification performance for training and validation data.
            
        :Input parameters:
            plt_range:          Range of training steps for which to plot performance (list)
            update:             Indicates, whether plot is to be updated or freshly created (boolean)
            
        """
        
        if plt_range: # extract prediction performance data for given plot range
                
            plot_data = np.array([self.training_perf[plt_range[0]:plt_range[1]], self.validation_perf[plt_range[0]:plt_range[1]]])
                
        else: # extract full prediction performance data on training and validation data
                
            plot_data = np.array([self.training_perf, self.validation_perf])
        
        plot_data = plot_data.T
                
        if not update: # if no prediction performance plot is created yet
            
            self.perf_fig = figure()
            perf_axes = self.perf_fig.add_subplot(111)            
            perf_axes.set_autoscale_on(True)
            title('Classification Performance')
            xlabel('Training step')
            ylabel('Loss')
            #show(block=False)
            
        else: # else update prediction performance plot
            
            perf_axes = self.perf_fig.axes[0]
            perf_axes.clear()
            
        perf_axes.plot(plot_data) 
        perf_axes.relim()
        perf_axes.autoscale_view(True,True,True)   
        self.perf_fig.canvas.draw()
        
        
    def plot_weights(self, target_layer, new_shape = [], update = False, remove_bias = False):
        """
        
        :Description:
            Creates plot of the weights of a certain layer, with a certain shape.
            
        :Input parameters:
            target_layer:       Indice of network layer, for which to plot weights (int)
            new_shape:          New shape, which the weights are transformed into before plotting (list)
            update:             Indicates, whether plot is to be updated or freshly created (boolean)
            remove_bias:        If true, the bias weights will be removed before plotting (boolean)
            
        """
        
        n = self.current_weights[target_layer].shape[1]
        
        if not update: # if no weights figure has been created yet
            
            self.weight_fig = figure()
            #show(block=False)
        
        if any(new_shape): # if weights are to be reshaped
            
            n1 = np.sqrt(n) # number of subplot rows
            n2 = n1 if n1**2 == n else n1+1 # number of subplot columns
                    
            for i in range(n): # go through all neurons of target layer
                
                weights = self.current_weights[target_layer][1:,i] if remove_bias else self.current_weights[target_layer][:,i] # extract weights
                weights = np.reshape(weights, [new_shape[0],new_shape[1]], order = 'C') # reshape weights
            
                if not update: # if no weights figure has been created yet, add new subplot
            
                    weight_axes = self.weight_fig.add_subplot(n1,n2,i+1)
                    
                else: # else update existing subplot
                    
                    weight_axes = self.weight_fig.axes[i]
                
                weight_axes.matshow(weights, cmap = 'coolwarm')  # plot weights 
                self.weight_fig.canvas.draw()
                
        else:
            
            n1 = n # number of subplot rows
            n2 = 1 # number of subplot columns
                    
            for i in range(n): # go through all neurons of target layer
                
                weights = self.current_weights[target_layer][1:,i] if remove_bias else self.current_weights[target_layer][:,i] # extract weights
                weights = np.reshape(weights,[1,len(self.current_weights[target_layer][:,i])], order = 'C')
            
                if not update: # if no weights figure has been created yet, add new subplot
            
                    weight_axes = self.weight_fig.add_subplot(n,1,i+1)
                    show(block=False)
                    
                else: # else update existing subplot
                    
                    weight_axes = self.weight_fig.axes[i]
                    
                weight_axes.matshow(weights, cmap = 'Greys', aspect = 'auto') # plot weights
                self.weight_fig.canvas.draw()
        

            
