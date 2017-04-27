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
import numpy as np
import cv2
import math

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

for idx,t_img in enumerate(x_test):
	cur_frame = x_test[idx].copy()
	pred_x = 10
	pred_y = 10
	pred_z = 0
	net_x = 0
	net_y = 0
	net_z = 1

	while(pred_z < 50):
		# Center the object
		while((abs(pred_x)+abs(pred_y)) > 0):
			prediction = loaded_model.predict(np.expand_dims(cur_frame, axis=0))[0]
			pred_x = int(prediction[0])
			pred_y = int(prediction[1])
			pred_z = int(prediction[2])
			print(str(pred_x) + '\t' + str(pred_y) + '\t' + str(pred_z) + '\t' + str(net_x) + '\t' + str(net_y) + '\t' + str(net_z)) 

			# Translate the image towards the object in question
			trans = np.float32([[1,0,-pred_x],[0,1,-pred_y]])
			cur_frame = cv2.warpAffine(cur_frame, trans,(500,500))
			net_x = net_x + pred_x*(float(1)/net_z)
			net_y = net_y + pred_y*(float(1)/net_z)

		# Zoom in 
		prediction = loaded_model.predict(np.expand_dims(cur_frame, axis=0))[0]
		pred_x = int(prediction[0])
		pred_y = int(prediction[1])
		pred_z = int(prediction[2])
		print(str(pred_x) + '\t' + str(pred_y) + '\t' + str(pred_z) + '\t' + str(net_x) + '\t' + str(net_y) + '\t' + str(net_z)) 
		# Zoom the image
		cur_frame = cv2.resize(cur_frame, (600,600), interpolation = cv2.INTER_CUBIC)
		cur_frame = cur_frame[50:550,50:550]
		net_z = net_z*float(6)/5

	bb_img = x_test[idx].copy()
	#cv2.circle(bb_img, (int(net_x+250), int(net_y+250)), 5, (0,100,0))
	width = float(500)/(net_z)
	height = float(500)/(net_z)
	cv2.rectangle(bb_img, (int(net_x+250-width/2), int(net_y+250-height/2)), (int(net_x+250+width/2), int(net_y+250+height/2)), (0,255,0)) 
	cv2.imshow('Original Image', x_test[idx])
	cv2.imshow('Bounding Box Image', bb_img)
	cv2.imshow('Zoomed Image', cur_frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
		
