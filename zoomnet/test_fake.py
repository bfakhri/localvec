from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from fake_data import vec
from keras.utils import plot_model
import cv2


#batch_size = 256
batch_size = 16 
#epochs = 6
epochs = 4
#epochs = 1

activation_fnc = 'relu'
#activation_fnc = 'tanh'
#activation_fnc = 'sigmoid'

# the data, shuffled and split between train and test sets
v = vec()
(x_train, y_train), (x_trainval, y_trainval), (x_test, y_test) = v.load_messy()
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
model.add(Dense(128, activation=activation_fnc))
model.add(Dropout(0.5))
model.add(Dense(64, activation=activation_fnc))
model.add(Dropout(0.5))
#model.add(Dense(32, activation=activation_fnc))
#model.add(Dropout(0.5))
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

model.save_weights('model.h5')
print('Saved Model as model.h5')

score = model.evaluate(x_test[1:30,:,:,:], y_test[1:30,:], verbose=0)

predictions = model.predict(x_test[1:15,:,:,:])
diff = y_test[1:15] - predictions
print("----------Predictions-------------")
print(predictions)
print("----------GroundTruth-------------")
print(y_test[1:15])
print("----------Diff-------------")
print(diff)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

for idx,pred in enumerate(predictions):
	disp_img = x_test[idx]
	pred_x = int(250*predictions[idx, 0]+250)
	pred_y = int(250*predictions[idx, 1]+250)
	pred_z = int(250*predictions[idx, 1]*100)+1
	gt_x = y_test[idx, 0]
	gt_y = y_test[idx, 1]
	cv2.circle(disp_img, (pred_x, pred_y), 10, (100,255,100), -1)
	cv2.line(disp_img, (250, 250), (pred_x, pred_y), (0,5,10), 4)
	#cv2.line(disp_img, (250, 250), (gt_x, gt_y), (100,5,10), 4)
	#cv2.line(temp_img, (cent_x, cent_y), (img_cent_x, img_cent_y), (200,255,150), lin_s)	
	
	cv2.imwrite('./temp/'+str(idx)+'output.jpg', disp_img)
	cv2.imshow('Centered Image', disp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
