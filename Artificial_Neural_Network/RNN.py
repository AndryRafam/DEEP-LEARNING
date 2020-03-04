""" RNN (Recurent Neural Network) on MNIST - 97,9% accuracy over 20 epochs"""

### import libraries
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from keras.layers import Dense, Activation 
from keras.layers import Input, SimpleRNN
from keras.regularizers import l2
from keras.models import Model
from keras.datasets import mnist


### load data
(x_train,y_train), (x_test,y_test) = mnist.load_data()


### compute the number of labels
num_labels = len(np.unique(y_train))


### reshape and renormalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size])
x_test = np.reshape(x_test,[-1,image_size,image_size])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


### network parameters
input_shape = (image_size,image_size)
batch_size = 128
units = 256
regul = l2(0.0001)
epochs = 20


### functional API to build CNN layers
inputs = Input(shape=input_shape)
def build_model():
	y = inputs
	y = SimpleRNN(units=units, kernel_regularizer=regul)(y)
	outputs = Dense(num_labels,activation='softmax')(y)
	return Model(inputs,outputs)

### build the model
model = build_model()
model.summary()

### train the model
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',
		    metrics=['accuracy'])
model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=epochs,batch_size=batch_size)
print("\n")

### evaluate the model
score = model.evaluate(x_test,y_test,batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100*score[1]))
print("\n")
