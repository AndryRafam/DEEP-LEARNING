### import libraries
import numpy as np
from keras.layers import Dense, Activation, Input, SimpleRNN
from keras.models import Model
from keras.datasets import mnist
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
dropout = 0.2
epochs = 20


### functional API to build CNN layers
inputs = Input(shape=input_shape)
y = SimpleRNN(units=units,dropout=dropout)(inputs)

### outputs
outputs = Dense(num_labels,activation='softmax')(y)


### build the model
model = Model(inputs=inputs, outputs=outputs)
model.summary()


### train the model
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',
		    metrics=['accuracy'])
history = model.fit(x_train,y_train,
                    validation_data=(x_test,y_test),
                    epochs=20,batch_size=batch_size)
print("\n")
print(history.history.keys())


### evaluate the model
score = model.evaluate(x_test,y_test,batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100*score[1]))
print("\n")


### plotting
plt.plot(history.history['accuracy'],'o',label='accuracy')
plt.plot(history.history['loss'],'o',label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend(loc='best')
plt.show()
