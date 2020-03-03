""" MLP on fashion_MNIST with l2 regularizer - 88,7% accuracy over 50 epochs """

### import libraries
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from keras.layers import Dense, Activation
from keras.layers import Input
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.regularizers import l2
import matplotlib.pyplot as plt

### load data
(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

### computer the number of labels
num_labels = len(np.unique(y_train))

### sample and plot 25 mnist digits from train dataset
indexes = np.random.randint(0,x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5,5,i+1)
    image = images[i]
    plt.imshow(image,cmap='gray')
    plt.axis('off')

plt.show()
plt.close('all')


### image dimensions (assumed square)
image_size = x_train.shape[1]
input_shape = image_size*image_size

### resize and normalize
x_train = np.reshape(x_train, [-1, input_shape])
x_train = x_train.astype('float32')/255
x_test = np.reshape(x_test, [-1, input_shape])
x_test = x_test.astype('float32')/255

### network parameters
batch_size = 128
hidden_units = 256
kernel_regul = l2(0.0001)
epochs=50

### 3-layer MLP model with relu and dropout after each layer

def build_model():
	inputs = Input(shape=(input_shape,))
	for i in range(2):
		y = inputs
		y = Dense(hidden_units,kernel_regularizer=kernel_regul)(y)
		y = Activation('relu')(y)
		
	y = Dense(num_labels)(y)
	outputs = Activation('softmax')(y)
	return Model(inputs,outputs)

model = build_model()
model.summary()

if __name__ == '__main__':
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	history = model.fit(x_train,y_train, epochs=epochs, validation_data=(x_test,y_test), batch_size=batch_size)
	print("\n")
	print(history.history.keys())

	### validate the model on test dataset to determmine generalization
	score = model.evaluate(x_test, y_test, batch_size=batch_size)
	print("\nTest accuracy: %.1f%%" % (100.0*score[1]))
	print("\n")
