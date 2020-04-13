""" CNN (Convolutional Neural Network) on fashion MNIST with BatchNormalization - 92,0% accuracy over 20 epochs """

# import libraries
from __future__ import print_function, division, absolute_import
import tensorflow
import numpy as np

from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist


class Init(object):
    def __init__(self):
        # load data
        (self.x_train,self.y_train), (self.x_test,self.y_test) = fashion_mnist.load_data()

        # compute the number of labels
        self.num_labels = len(np.unique(self.y_train))

        # reshape and normalize input images
        self.image_size = self.x_train.shape[1]
        self.x_train = np.reshape(self.x_train,[-1,self.image_size,self.image_size,1])
        self.x_test = np.reshape(self.x_test,[-1,self.image_size,self.image_size,1])
        self.x_train = self.x_train.astype('float32')/255
        self.x_test = self.x_test.astype('float32')/255

        # network parameters
        self.input_shape = (self.image_size,self.image_size,1)
        self.batch_size = 128
        self.kernel_size = 3
        self.filters = 64
        self.dropout = 0.3
        self.epochs = 20

class CNN(Init):
    def build_train_model(self):
        # build model
        self.inputs = Input(shape=self.input_shape)
        self.y = self.inputs
        for i in range(3):
            self.y = Conv2D(filters=self.filters,kernel_size=self.kernel_size,padding='same')(self.y)
            self.y = BatchNormalization()(self.y)
            self.y = Activation('relu')(self.y)
            self.y = MaxPooling2D()(self.y)
        self.y = Flatten()(self.y)
        self.y = Dropout(self.dropout)(self.y)
        self.outputs = Dense(self.num_labels,activation='softmax')(self.y)
        self.model = Model(self.inputs,self.outputs)

        self.model.summary()

        # train model
        self.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(self.x_train,self.y_train,validation_data=(self.x_test,self.y_test),epochs=self.epochs,
                       batch_size=self.batch_size)
        self.score = self.model.evaluate(self.x_test,self.y_test,batch_size=self.batch_size)
        print("\nTest accuracy: %.1f%%" % (100*self.score[1]))


if __name__=='__main__':
    cnn = CNN()
    cnn.build_train_model()
