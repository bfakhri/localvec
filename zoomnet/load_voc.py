# Converts the bounding boxes to vectors from the center

import cv2
# To get filenames in directory
import glob
# To store the files into a big array
import numpy as np

def voc_files():
	img_files = glob.glob("../data/VOC2012/JPEGImages/*.jpg")
	n_total_files = len(img_files)
	n_train_files = n_total_files*9/10
	n_test_files = n_total_files - n_train_files
	print("Number of total image files: " + str(n_total_files))
	print("Number of train image files: " + str(n_train_files))
	print("Number of test image files: " + str(n_test_files))

	# Create np array for train and test files
	x_train = np.empty([n_train_files, 3, 38, 38], dtype=np.int8, order='C')
	x_test = np.empty([n_test_files, 3, 38, 38], dtype=np.int8, order='C')
	

	for f in img_files:
		# Load an color image 
		img = cv2.imread(f)
		print type(img)
		print img.shape

	

voc_files()
