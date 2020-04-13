from __future__ import print_function, absolute_import, division
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import fashion_mnist

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)


class Init(object):
    def __init__(self):
        # load dataset
        (self.x_train,self.y_train), (self.x_test,self.y_test) = fashion_mnist.load_data()

        # reshape and normalize
        self.image_size = self.x_train.shape[1]
        self.x_train = np.reshape(self.x_train, [-1,self.image_size,self.image_size,1])
        self.x_test = np.reshape(self.x_test, [-1,self.image_size,self.image_size,1])
        self.x_train = self.x_train.astype('float32')/255
        self.x_test = self.x_test.astype('float32')/255

        # generate corrupted MNIST images by adding noise with normal dist
        # centered at 0.5 and std=0.5
        self.noise = np.random.normal(loc=0.5, scale=0.5, size=self.x_train.shape)
        self.x_train_noisy = self.x_train + self.noise
        self.noise = np.random.normal(loc=0.5, scale=0.5, size=self.x_test.shape)
        self.x_test_noisy = self.x_test + self.noise

        # clip pixel values >1.0 to 1.0 and <0.0 to 0.0 (adding noise may exceed normalized pixel values)
        self.x_train_noisy = np.clip(self.x_train_noisy, 0. , 1.)
        self.x_test_noisy = np.clip(self.x_test_noisy, 0. , 1.)

        # network parameters
        self.input_shape = (self.image_size,self.image_size,1)
        self.batch_size = 32
        self.kernel_size = 3
        self.latent_dim = 16

        # encoder/decoder number of CNN layers and filters per layer
        self.layer_filters = [32,64]


class DENOISED(Init):
    # build the autoencoder model
    def auto_encoder(self):
        # build the encoder
        self.inputs = Input(shape=self.input_shape, name='encoder_input')
        self.x = self.inputs

        for filters in self.layer_filters:
            self.x = Conv2D(filters=filters, kernel_size=self.kernel_size,
                            strides=2, activation='relu', padding='same')(self.x)

        self.shape = K.int_shape(self.x)

        # generate the latent_vector
        self.x = Flatten()(self.x)
        self.latent = Dense(self.latent_dim, name='latent_vector')(self.x)

        # instantiate encoder model
        self.encoder = Model(self.inputs, self.latent, name='encoder')
        self.encoder.summary()

        # build the decoder model
        self.latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        self.x = Dense(self.shape[1]*self.shape[2]*self.shape[3])(self.latent_inputs)
        self.x = Reshape((self.shape[1], self.shape[2], self.shape[3]))(self.x)

        for filters in self.layer_filters[::-1]:
            self.x = Conv2DTranspose(filters=filters, kernel_size=self.kernel_size,
                                     strides=2, activation='relu', padding='same')(self.x)

        # reconstruct the denoised input
        self.outputs = Conv2DTranspose(filters=1, kernel_size=self.kernel_size,
                                       padding='same', activation='sigmoid',
                                       name='decoder_output')(self.x)

        # instantiate decoder model
        self.decoder = Model(self.latent_inputs, self.outputs, name='decoder')
        self.decoder.summary()

        # autoencoder = encoder+decoder
        self.autoencoder = Model(self.inputs, self.decoder(self.encoder(self.inputs)), name='autoencoder')
        self.autoencoder.summary()

        # MSE loss function, Adam optimizer
        self.autoencoder.compile(loss='mse', optimizer='adam')

        # train the autoencoder
        self.autoencoder.fit(self.x_train_noisy, self.x_train_noisy,
                             validation_data=(self.x_test_noisy, self.x_test_noisy),
                             epochs=10, batch_size=self.batch_size)

        # predict the autoencoder output from corrupted test images
        self.x_decoded = self.autoencoder.predict(self.x_test_noisy)


    def display(self):
        self.rows, self.cols = 3, 9
        self.num = self.rows*self.cols
        self.imgs = np.concatenate([self.x_test[:self.num], self.x_test_noisy[:self.num], self.x_decoded[:self.num]])
        self.imgs = self.imgs.reshape((self.rows*3, self.cols, self.image_size, self.image_size))
        self.imgs = np.vstack(np.split(self.imgs, self.rows, axis=1))
        self.imgs = self.imgs.reshape((self.rows*3, -1, self.image_size, self.image_size))
        self.imgs = np.vstack([np.hstack(i) for i in self.imgs])
        self.imgs = (self.imgs*255).astype(np.uint8)
        plt.figure()
        plt.axis('off')
        plt.title('Original images: top rows, '
                  'Corrupted Input: middle rows '
                  'Denoised Input: third rows')
        plt.imshow(self.imgs, interpolation='none', cmap='gray')
        Image.fromarray(self.imgs).save('corrupted_and_denoised.png')
        plt.show()



# main program
if __name__=='__main__':
    D = DENOISED()
    D.auto_encoder()
    D.display()
