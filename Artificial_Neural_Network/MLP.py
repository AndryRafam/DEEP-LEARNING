""" MLP (Multi Layer Perceptron) on fashion MNIST with l2 regularizer - 88,7% accuracy over 50 epochs """

# import libraries
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.regularizers import l2


class Init(object):
    def __init__(self): # constructor
        #load_data
        (self.x_train,self.y_train), (self.x_test,self.y_test) = fashion_mnist.load_data()

        # compute the number of labels
        self.num_labels = len(np.unique(self.y_train))

        # image dimensions (assumed square)
        self.image_size = self.x_train.shape[1]
        self.input_shape = self.image_size*self.image_size

        # resize and normalize
        self.x_train = np.reshape(self.x_train, [-1,self.input_shape])
        self.x_train = self.x_train.astype('float32')/255
        self.x_test = np.reshape(self.x_test, [-1,self.input_shape])
        self.x_test = self.x_test.astype('float32')/255

        # network parameters
        self.batch_size = 128
        self.hidden_units = 256
        self.kernel_regul = l2(0.0001)
        self.epochs = 50

class MLP(Init):
    def build_train_model(self):
        # build model
        self.inputs = Input(shape=(self.input_shape,))
        for i in range(2):
            self.y = self.inputs
            self.y = Dense(self.hidden_units,kernel_regularizer=self.kernel_regul)(self.y)
            self.y = Activation('relu')(self.y)

        self.y = Dense(self.num_labels)(self.y)
        self.outputs = Activation('softmax')(self.y)
        self.model = Model(self.inputs,self.outputs)

        self.model.summary()

        # train model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train,self.y_train, epochs=self.epochs, validation_data=(self.x_test,self.y_test),
                       batch_size=self.batch_size)
        self.score = self.model.evaluate(self.x_test,self.y_test,batch_size=self.batch_size)
        print("\nTest accuracy: %.1f%%" % (100.0*self.score[1]))
        print("\n")


if __name__ =='__main__':
    mlp = MLP()
    mlp.build_train_model()
