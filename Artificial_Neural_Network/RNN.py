""" RNN (Recurent Neural Network) on fashion MNIST - 88,3% over 50 epochs """

# import libraries
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow
import numpy as np

from tensorflow.keras.layers import Dense, Activation, Input, SimpleRNN
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist


class Init(object):
    def __init__(self): # constructor
        # load data
        (self.x_train,self.y_train), (self.x_test,self.y_test) = fashion_mnist.load_data()

        # compute the number of labels
        self.num_labels = len(np.unique(self.y_train))

        # reshape and renormalize input images
        self.image_size = self.x_train.shape[1]
        self.x_train = np.reshape(self.x_train,[-1,self.image_size,self.image_size])
        self.x_test = np.reshape(self.x_test,[-1,self.image_size,self.image_size])
        self.x_train = self.x_train.astype('float32')/255
        self.x_test = self.x_test.astype('float32')/255

        # network parameters
        self.input_shape = (self.image_size,self.image_size)
        self.batch_size = 128
        self.units = 256
        self.regul = l2(0.0001)
        self.epochs = 50

class RNN(Init):
    def build_train_model(self):
        #build model
        self.inputs = Input(shape=self.input_shape)
        self.y = SimpleRNN(units=self.units,kernel_regularizer=self.regul)(self.inputs)
        self.outputs = Dense(self.num_labels,activation='softmax')(self.y)
        self.model = Model(self.inputs,self.outputs)

        self.model.summary()
        print("\n")

        # train model
        self.model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(self.x_train,self.y_train,validation_data=(self.x_test,self.y_test),
                       epochs=self.epochs,batch_size=self.batch_size)
        self.score = self.model.evaluate(self.x_test,self.y_test,batch_size=self.batch_size)
        print("\nTest accuracy: %.1f%%" % (100*self.score[1]))
        print("\n")



if __name__=='__main__':
    rnn = RNN()
    rnn.build_train_model()
