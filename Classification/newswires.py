# Reuters newswire classification - 80.5 % accuracy over 5 epochs

from __future__ import print_function
from __future__ import absolute_import

from keras.datasets import reuters
from keras.models import Model
from keras.layers import Dense, Input, Activation
from keras.optimizers import RMSprop
from keras.layers.merge import concatenate

import numpy as np

### downloading the data
(train_data,train_labels), (test_data,test_labels) = reuters.load_data(num_words=10000)


### Encoding the data
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.array(train_labels)
y_test = np.array(test_labels)


### network parameters
hidden_units = 512
epochs = 5
batch_size = 32
input_shape = (10000,)
num_labels = len(np.unique(y_train))

### build the model
def build_model():
	left_inputs = Input(shape=input_shape)
	x = left_inputs
	for i in range(2):
		x = Dense(hidden_units)(x)
		x = Activation('relu')(x)
	
	right_inputs = Input(shape=input_shape)
	y = right_inputs
	for i in range(2):
		y = Dense(hidden_units)(y)
		y = Activation('relu')(y)
	
	z = concatenate([x,y])
	outputs = Dense(num_labels)(z)
	outputs = Activation('softmax')(outputs)
	return Model([left_inputs,right_inputs],outputs)

### Compiling and training the model
model = build_model()
optimizer = RMSprop(lr=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
	      metrics=['accuracy'])


x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]


history = model.fit([partial_x_train,partial_x_train],
                    partial_y_train,
                    batch_size=batch_size,
                    validation_data=([x_val,x_val],y_val),
                    epochs=epochs)

score = model.evaluate([x_test,x_test],y_test)
print("\nScore: %.1f%%" % (100.0*score[1]))
