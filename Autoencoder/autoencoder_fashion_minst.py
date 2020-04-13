from __future__ import print_function, division, absolute_import

import tensorflow
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt


class Init(object):
    def __init__(self):
        # load dataset
        (self.x_train,self.y_train), (self.x_test,self.y_test) = fashion_mnist.load_data()

        # reshape and normalize
        self.image_size = self.x_train.shape[1]
        self.x_train = np.reshape(self.x_train,[-1,self.image_size,self.image_size,1])
        self.x_test = np.reshape(self.x_test,[-1,self.image_size,self.image_size,1])
        self.x_train = self.x_train.astype('float32')/255
        self.x_test = self.x_test.astype('float32')/255

        # network parameters
        self.input_shape = (self.image_size,self.image_size,1)
        self.batch_size = 32
        self.kernel_size = 3
        self.latent_dim = 16

        # encoder/decoder number of filters per CNN layer
        self.layer_filters = [32,64]

class Autoencoder(Init):
    # build the autoencoder model
    def auto_encoder(self):
        #build the encoder
        self.inputs = Input(shape=self.input_shape, name='encoder_input')
        self.x = self.inputs
        for filters in self.layer_filters:
            self.x = Conv2D(filters=filters, kernel_size=self.kernel_size,
                            activation='relu', strides=2,
                            padding='same')(self.x)

        self.shape = K.int_shape(self.x)

        # generate latent vector
        self.x = Flatten()(self.x)
        self.latent = Dense(self.latent_dim, name='latent_vector')(self.x)

        # instantiate the encoder model
        self.encoder = Model(self.inputs, self.latent, name='encoder')
        self.encoder.summary()
        plot_model(self.encoder, to_file='encoder.png',show_shapes=True)


        # build the decoder
        self.latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        self.x = Dense(self.shape[1]*self.shape[2]*self.shape[3])(self.latent_inputs)
        self.x = Reshape((self.shape[1],self.shape[2],self.shape[3]))(self.x)

        for filters in self.layer_filters[::-1]:
            self.x = Conv2DTranspose(filters=filters, kernel_size=self.kernel_size,
                                   activation='relu', strides=2,
                                   padding='same')(self.x)

        # reconstruct the input
        self.outputs = Conv2DTranspose(filters=1, kernel_size=self.kernel_size,
                                       activation='sigmoid', padding='same',
                                       name='decoder_output')(self.x)

        # instantiate decoder model
        self.decoder = Model(self.latent_inputs, self.outputs, name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, to_file='decoder.png', show_shapes=True)


        # instantiate the autoencoder model
        self.autoencoder = Model(self.inputs, self.decoder(self.encoder(self.inputs)),
                                 name='autoencoder')
        self.autoencoder.summary()
        plot_model(self.autoencoder, to_file='autoencoder.png',
                   show_shapes=True)

        # MSE loss function, Adam optimizer
        self.autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # train the autoencoder
        self.autoencoder.fit(self.x_train, self.x_train, validation_data=(self.x_test,self.x_test),
                             epochs=1, batch_size=self.batch_size)

        # predict the autoencoder output from test data
        self.x_decoded = self.autoencoder.predict(self.x_test)


    def display(self):
        # display the 1st 8 test input and decoded images
        self.imgs = np.concatenate([self.x_test[:8], self.x_decoded[:8]])
        self.imgs = self.imgs.reshape((4,4,self.image_size,self.image_size))
        self.imgs = np.vstack([np.hstack(i) for i in self.imgs])
        plt.figure()
        plt.axis('off')
        plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
        plt.imshow(self.imgs, interpolation='none', cmap='gray')
        plt.savefig('input_and_decoded.png')
        plt.show()


# main program
if __name__=='__main__':
    at = Autoencoder()
    at.auto_encoder()
    at.display()
