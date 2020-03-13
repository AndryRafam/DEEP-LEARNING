"""Fashion Mnist with CNN (Convolutional Neural Network) - 90.9% accuracy over 20 epochs"""

# import libraries
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dropout, Dense, Input, BatchNormalization
from tensorflow.keras.datasets import fashion_mnist


class CNN():
	def __init__(self):
		# loading the data
		(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		
		# reshape and normalize
		self.num_labels = len(np.unique(self.y_train))
		self.image_size = self.x_train.shape[1]
		self.x_train = np.reshape(self.x_train,[-1,self.image_size,self.image_size,1])
		self.x_test = np.reshape(self.x_test,[-1,self.image_size,self.image_size,1])
		self.x_train = self.x_train.astype('float32')/255
		self.x_test = self.x_test.astype('float32')/255
		
		# network parameters
		self.input_shape = (self.image_size,self.image_size,1)
		self.batch_size = 32
		self.dropout = 0.25
		self.filters = 32
		self.epochs = 20
		self.kernel_size = 5
		
	def build_model(self,filters,kernel_size,dropout):
		self.filters = filters
		self.kernel_size = kernel_size
		self.dropout = dropout
		self.inputs = Input(shape=self.input_shape)
		for i in range(3):
			self.cnn = Conv2D(self.filters,self.kernel_size,padding='same', activation='relu')(self.inputs)
			self.cnn = BatchNormalization()(self.cnn)
			self.cnn = Dropout(self.dropout)(self.cnn)
			self.cnn = MaxPooling2D()(self.cnn)
			self.filters *= 2
		self.cnn = Flatten()(self.cnn)
		self.cnn = Dropout(dropout)(self.cnn)
		self.outputs = Dense(self.num_labels, activation='softmax')(self.cnn)
		return Model(self.inputs,self.outputs)


	def train_model(self):
		self.model = self.build_model(self.filters,self.kernel_size,self.dropout)
		self.model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		self.model.fit(self.x_train,self.y_train, epochs=self.epochs, batch_size=self.batch_size)
			
		self.score = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
		print("\nACCURACY: %.1f%%" % (100*self.score[1]))

if __name__ == '__main__':
	cnn = CNN()
	cnn.train_model()
