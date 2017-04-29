from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from data_gen import load_synth 


batch_size = 16 
epochs = 4

activation_fnc = 'relu'

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_trainval, y_trainval), (x_test, y_test) = load_synth()
cols = x_train.shape[1]
rows = x_train.shape[2]
in_shape = (cols, rows, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_trainval.shape[0], 'trainval samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5),
                 activation=activation_fnc,
                 input_shape=in_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation=activation_fnc,
                 input_shape=in_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation=activation_fnc,
                 input_shape=in_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation=activation_fnc))
model.add(Dropout(0.5))
model.add(Dense(64, activation=activation_fnc))
model.add(Dropout(0.5))
model.add(Dense(32, activation=activation_fnc))
model.add(Dropout(0.5))
model.add(Dense(3, activation='linear'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_trainval, y_trainval))

# Save model params to JSON file
json_model = model.to_json()
with open("model_params.json", "w") as json_file:
    json_file.write(json_model)
# Save weights to HDF5 file
model.save_weights("model_weights.h5")
print("Saved model to disk")
