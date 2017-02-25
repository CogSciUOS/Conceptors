# -*- coding: utf-8 -*-
"""
@author: kstandvoss
"""

from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from hyperopt import hp, fmin, tpe

import numpy as np
import os
from matplotlib import pyplot as plt

class DeepRNN:
    """    
    :Description:
        Songclassifier using a GRU neural network
    """

    #Parameters to optimize
    space = ( 
        hp.uniform( 'dropout', 0, 0.6),
        hp.choice( 'neurons', [50, 100, 200]),
        hp.choice( 'activation', ['relu', 'tanh', 'sigmoid']),
        hp.choice( 'layers', [1, 2, 3]),
        hp.choice( 'epochs', [1,3,5,10]),
        hp.choice( 'batch_size', [8,16,32,64,128])
    )

    def __init__(self,dropout, neurons, activation, layers, epochs, batch_size, max_len, n_songs):
        """
        
        :Description:
            Initializes instance of DeepRNN.
        
        :Input parameters:
            dropout:    Amount of dropout
            neurons:    Number of neurons per layer
            layers:     Number of layers
            activation: Activation function
            epochs:     Number of epochs
            batch_size: Size of mini-batches
            embedding:  Length of embedding vectors
            max_len:    Maximum song length
            n_songs:    Number of classes
            
        """
        self.dropout = dropout
        self.neurons = int(neurons)
        self.layers = layers
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size

        self.embedding = 512
        self.max_len = max_len
        self.n_songs = n_songs

        self.buildModel()
    

    def buildModel(self):
        '''
        :Description:
            Build neural network model
        '''

        self.model = Sequential()
        self.model.add(Embedding(self.embedding, 16, input_length=self.max_len))
        for l in range(self.layers-1):
            self.model.add(GRU(self.neurons, activation=self.activation, return_sequences=True, dropout_W=self.dropout, dropout_U=self.dropout))

        self.model.add(GRU(self.neurons, activation=self.activation, return_sequences=False, dropout_W=self.dropout, dropout_U=self.dropout))    
        
        self.model.add(Dense(self.n_songs))
        self.model.add(Activation('softmax'))            
        
    def train(self, X_train, Y_train, X_test, Y_test, noise=[0]):
        

        """
        :Description:
            Trains the network by going through the training data in random mini-batches for multiple epochs.
            
        :Input parameters:
            X_train:    Training Data
            Y_train:    Training Labels
            X_test:     Validation Data
            Y_test:     Validation Labels
            noise:      Noise levels to apply to data
             
        """

        """ compile """
        
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  
        
        
        """ training procedure """    

        
        modelPath = './checkpoints/models/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        modelCheck = ModelCheckpoint(modelPath, monitor='val_loss', verbose=0,
                                        save_best_only=True, mode='auto')   
       

        SNRs = np.random.choice(noise,X_train.shape[0])
        noiseLVL = np.sqrt(np.var(X_train,1) / SNRs)
        noise = noiseLVL[...,None]*np.random.randn(X_train.shape[0],X_train.shape[1])
        X_train += noise
        np.clip(X_train,-(self.embedding-1),(self.embedding-1), X_train)

        history = self.model.fit(X_train, Y_train, 
                                    batch_size=self.batch_size, nb_epoch=self.epochs, shuffle=True,
                                    validation_data=(X_test,Y_test), callbacks= [],verbose=2)

        score = self.model.evaluate(X_test,Y_test, verbose=0)

        return score[1]

    
    def evaluate(self, data, labels):
        """
        :Description:
            Evaluate network on test data
            
        :Input parameters:
            data:   Testdata
            labels: Testlabels
        """
        scores = self.model.evaluate(data, labels, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
         
        return scores[1]    
