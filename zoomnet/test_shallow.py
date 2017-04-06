from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from vec import vec
from keras.utils import plot_model


batch_size = 1 
epochs = 20

activation_fnc = 'relu'
#activation_fnc = 'tanh'
#activation_fnc = 'sigmoid'

# the data, shuffled and split between train and test sets
v = vec()
(x_train, y_train), (x_trainval, y_trainval), (x_test, y_test) = v.load_pure()
# FOR DEBUGGING
x_train = x_train[0:20, :, :, :]
y_train = y_train[0:20, :]
x_trainval = x_train
y_trainval = y_train
#x_train, y_train), (x_trainval, y_trainval), (x_test, y_test) = v.load_messy()
img_rows = x_train.shape[2]
img_cols = x_train.shape[1]

print(x_train.shape)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

print(x_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_trainval.shape[0], 'trainval samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5),
                 activation=activation_fnc,
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation=activation_fnc,
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation=activation_fnc,
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation=activation_fnc))
model.add(Dropout(0.5))
model.add(Dense(32, activation=activation_fnc))
model.add(Dropout(0.5))
model.add(Dense(3, activation='linear'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

plot_model(model, to_file='model.png', show_shapes='True')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_trainval, y_trainval))
score = model.evaluate(x_test[1:30,:,:,:], y_test[1:30,:], verbose=0)
print(model.predict(x_test[1:20,:,:,:]))
print(y_test[1:20])
print('Test loss:', score[0])
print('Test accuracy:', score[1])
