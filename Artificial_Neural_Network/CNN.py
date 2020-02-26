### import libraries
import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from random import seed, randint
import matplotlib.pyplot as plt

### load data
(x_train,y_train), (x_test,y_test) = mnist.load_data()


### compute the number of labels
num_labels = len(np.unique(y_train))


### verify the data
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


### reshape and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size,1])
x_test = np.reshape(x_test,[-1,image_size,image_size,1])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


### network parameters
input_shape = (image_size,image_size,1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3


### functional API to build CNN layers
inputs = Input(shape=input_shape)
for i in range(3):
	y = Conv2D(filters=filters,
			 kernel_size=kernel_size,
			 padding='same',
			 activation='relu')(inputs)
	y = MaxPooling2D()(y)


### image to vector before connecting to dense layer
y = Flatten()(y)


### dropout regularization
y = Dropout(dropout)(y)
outputs = Dense(num_labels,activation='softmax')(y)


### build the model by supplying inputs/outputs
model = Model(inputs=inputs, outputs=outputs)


### network model in text
model.summary()


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


### train the model
history = model.fit(x_train,y_train,
          	    validation_data=(x_test,y_test),
                   epochs=20,batch_size=batch_size)


### model accuracy
score = model.evaluate(x_test,y_test,batch_size=batch_size)
print("\nScore: %.1f%%" % (100*score[1]))


### plotting
plt.plot(history.history['accuracy'],'bo',label='accuracy')
plt.plot(history.history['loss'],'ro',label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0,2])
plt.legend(loc='best')
plt.show()


### prediction
image_index = randint(0,10000)
plt.imshow(x_test[image_index].reshape(28,28),cmap='gray')
plt.show()
pred = model.predict(x_test[image_index].reshape(-1,image_size,image_size,1))
print(pred.argmax())
