# import libraries
from numpy import loadtxt
from keras.models import Model
from keras.layers import Dense, Activation, Input, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#load the dataset
dataset = loadtxt("pima-indians-diabetes.csv",delimiter=',')

#split into train and test data
X = dataset[:,0:8]
Y = dataset[:,8]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

# networks parameters
epochs = 1000
batch_size = 100
input_shape = (8,)

# model
def build_model():
	inputs = Input(shape=input_shape)
	z = Dense(500,activation='relu')(inputs)
	z = Dense(100,activation='relu')(z)
	outputs = Dense(1,activation='sigmoid')(z)
	return Model(inputs,outputs)

model = build_model()

model.compile(loss='binary_crossentropy',optimizer='adamax',metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# plotting
plt.plot(history.history['accuracy'],'b', label='accuracy')
plt.plot(history.history['loss'], 'r', label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([-0.5,1.5])
plt.legend(loc='best')
plt.show()

score = model.evaluate(x_test,y_test)
print("\nAccuracy: %.1f%%" % (100*score[1]))
print(dataset.shape)
