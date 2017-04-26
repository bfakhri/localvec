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

loaded_model.load_weights("model.h5")
print("Loaded model from file model.h5")

loaded_model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy'])

score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
