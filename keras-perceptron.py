import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model = Sequential()
model.add(Dense(1, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(1, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(1, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(1, input_dim=100))
model.add(Activation('relu'))
model.compile(optimizer='sgd',
			  loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])

data = np.random.random((1000000, 100))
labels = np.random.randint(2, size=(1000000, 1))

model.fit(data, labels, epochs=100, batch_size=32)
