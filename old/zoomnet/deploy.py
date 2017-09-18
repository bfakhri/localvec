from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from fake_data import vec
from keras.utils import plot_model
from keras.models import model_from_json

import cv2

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights
loaded_model.load_weights("model.h5")
print("Loaded model from file model.h5")

loaded_model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy'])

# Load the test data
v = vec()
(x_train, y_train), (x_trainval, y_trainval), (x_test, y_test) = v.load_messy()

score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

predictions = loaded_model.predict(x_test)

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

