'''
-----------------------------------------------------------------------------
LICENSE
-----------------------------------------------------------------------------
The MIT License

Copyright (c) 2017 - Bijan Fakhri  

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
-----------------------------------------------------------------------------
FILE NOTES
-----------------------------------------------------------------------------
This file generates the data used to train the ZoomNet model. The primary 
method is load_synth() which generates synthetic data and returns it 
partitioned into 6 arrays, two for training (x is data, y are labels), two
for validation during training, and two for testing. 

The synthetic data is comprised of cocentric circles drawn on real images 
from the Pascal VOC2012 dataset.

The rest of the methods are either helper methods or were created during 
testing of the ZoomNet

Written by Bijan Fakhri
-----------------------------------------------------------------------------
'''
import numpy as np			# Array manipulation
import cv2				# Image manipulation
import xml.etree.ElementTree as ET	# To read XML docs
import random as rand

file_listdir = "../data/VOCdevkit/VOC2012/ImageSets/Main/"
img_listdir = "../data/VOCdevkit/VOC2012/JPEGImages/"
xml_listdir = "../data/VOCdevkit/VOC2012/Annotations/"
dest_dir = "./output/"

NUM_CHANS = 3
VECT_DIMS = 3

def populate_array(sample_list, MAX_X, MAX_Y, obj_type):
	# Create the output arrays for Keras
	x = np.zeros([len(sample_list), MAX_Y, MAX_X, NUM_CHANS], dtype=np.uint8)
	y = np.zeros([len(sample_list), VECT_DIMS], dtype=np.float32)
	# Populate those arrays	
	for idx,sample in enumerate(sample_list):
		frame = np.zeros([MAX_Y, MAX_X, NUM_CHANS], dtype=np.uint8)
		cent_x = rand.uniform(-0.4, 0.4)
		cent_y = rand.uniform(-0.4, 0.4)
		scale = rand.uniform(0.05, 0.4)
		# Load the image 
		img = cv2.imread(img_listdir+sample+".jpg")
		height, width, channels = img.shape
		# Write image to frame (500x500)
		frame[(MAX_Y/2-height/2):(MAX_Y/2-height/2+height), (MAX_X/2-width/2):(MAX_X/2-width/2+width), :] = img
		# Draw circles on the image
		circle_x = int(cent_x*MAX_X) + MAX_X/2
		circle_y = int(cent_y*MAX_Y) + MAX_Y/2
		circle_s = int(scale*MAX_X/2)
		#print circle_x, circle_y, circle_s
		#print cent_x, cent_y, scale
		cv2.circle(frame,(circle_x, circle_y), circle_s, (255,0,0), -1)
		cv2.circle(frame,(circle_x, circle_y), int(circle_s*0.9), (0,255,255), -1)
		cv2.circle(frame,(circle_x, circle_y), int(circle_s*0.7), (255,255,255), -1)
		cv2.circle(frame,(circle_x, circle_y), int(circle_s*0.5), (0,0,0), -1)
		# Write to output array
		x[idx] = frame

		#cv2.imshow('Centered Image', frame)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		# Write out to file
		cv2.imwrite(dest_dir+str(idx)+'_fake.jpg', frame)
		# Make the labels
		y[idx, 0] = cent_x
		y[idx, 1] = cent_y
		y[idx, 2] = scale

	return (x, y)

def populate_array_binary(sample_list, MAX_X, MAX_Y, obj_type):
	# Create the output arrays for Keras
	x = np.zeros([len(sample_list), MAX_Y, MAX_X, NUM_CHANS], dtype=np.uint8)
	y = np.zeros([len(sample_list), 1], dtype=np.float32)
	# Populate those arrays	
	for idx,sample in enumerate(sample_list):
		frame = np.zeros([MAX_Y, MAX_X, NUM_CHANS], dtype=np.uint8)
		cent_x = rand.uniform(-0.5, 0.5)
		cent_y = rand.uniform(-0.5, 0.5)
		scale = rand.uniform(0.05, 0.7)
		# Load the image 
		img = cv2.imread(img_listdir+sample+".jpg")
		height, width, channels = img.shape
		# Write image to frame (500x500)
		frame[(MAX_Y/2-height/2):(MAX_Y/2-height/2+height), (MAX_X/2-width/2):(MAX_X/2-width/2+width), :] = img
		# Draw circles on the image
		circle_x = int(cent_x*MAX_X) + MAX_X/2
		circle_y = int(cent_y*MAX_Y) + MAX_Y/2
		circle_s = int(scale*MAX_X/2)
		#print circle_x, circle_y, circle_s
		#print cent_x, cent_y, scale
		cv2.circle(frame,(circle_x, circle_y), circle_s, (255,0,0), -1)
		cv2.circle(frame,(circle_x, circle_y), int(circle_s*0.9), (0,255,255), -1)
		cv2.circle(frame,(circle_x, circle_y), int(circle_s*0.7), (255,255,255), -1)
		cv2.circle(frame,(circle_x, circle_y), int(circle_s*0.5), (0,0,0), -1)
		# Write to output array
		x[idx] = frame
		#cv2.imshow('Centered Image', frame)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		# Write out to file
		#cv2.imwrite(dest_dir+str(idx)+'_fake.jpg', frame)
		# Make the labels
		y[idx, 0] = cent_x
		y[idx, 1] = cent_y
		y[idx, 2] = scale

	return (x, y)


# Training array has samples with birds and without birds present
def load_synth(obj_type="bird"):
	# Get handle to list of files 
	trn_str = str(file_listdir)+str(obj_type)+str("_train.txt")
	trnval_str = str(file_listdir)+str(obj_type)+str("_trainval.txt")
	val_str = str(file_listdir)+str(obj_type)+str("_val.txt")
	print("Opening dataset with type: "+str(obj_type))
	print("Opening "+trn_str)
	print("Opening "+trnval_str)
	print("Opening "+val_str)
	train_file_hdl = open(trn_str, "r")
	trainval_file_hdl = open(trnval_str, "r")
	val_file_hdl = open(val_str, "r")

	train_list = []
	trainval_list = []
	val_list = []
	# Put into vectors
	for line in train_file_hdl:
		spl = line.split()
		#if(spl[1] == '1'):
		train_list.append(spl[0])
	for line in trainval_file_hdl:
		spl = line.split()
		if(spl[1] == '1'):
			trainval_list.append(spl[0])
	for line in val_file_hdl:
		spl = line.split()
		if(spl[1] == '1'):
			val_list.append(spl[0])

	print("There are : " + str(len(train_list)) + " training samples")
	print("There are : " + str(len(trainval_list)) + " training validation samples")
	print("There are : " + str(len(val_list)) + " validation samples")

	MAX_X = 500
	MAX_Y = 500

	return populate_array(train_list, MAX_X, MAX_Y, obj_type), populate_array(trainval_list, MAX_X, MAX_Y, obj_type), populate_array(val_list, MAX_X, MAX_Y, obj_type)

# Training array has samples with birds and without birds present
def load_messy_binary(obj_type="bird"):
	# Get handle to list of files 
	trn_str = str(file_listdir)+str(obj_type)+str("_train.txt")
	trnval_str = str(file_listdir)+str(obj_type)+str("_trainval.txt")
	val_str = str(file_listdir)+str(obj_type)+str("_val.txt")
	print("Opening dataset with type: "+str(obj_type))
	print("Opening "+trn_str)
	print("Opening "+trnval_str)
	print("Opening "+val_str)
	train_file_hdl = open(trn_str, "r")
	trainval_file_hdl = open(trnval_str, "r")
	val_file_hdl = open(val_str, "r")

	train_list = []
	trainval_list = []
	val_list = []
	# Put into vectors
	for line in train_file_hdl:
		spl = line.split()
		#if(spl[1] == '1'):
		train_list.append(spl[0])
	for line in trainval_file_hdl:
		spl = line.split()
		if(spl[1] == '1'):
			trainval_list.append(spl[0])
	for line in val_file_hdl:
		spl = line.split()
		if(spl[1] == '1'):
			val_list.append(spl[0])

	print("There are : " + str(len(train_list)) + " training samples")
	print("There are : " + str(len(trainval_list)) + " training validation samples")
	print("There are : " + str(len(val_list)) + " validation samples")

	MAX_X = 500
	MAX_Y = 500

	return populate_array_binary(train_list, MAX_X, MAX_Y, obj_type), populate_array(trainval_list, MAX_X, MAX_Y, obj_type), populate_array(val_list, MAX_X, MAX_Y, obj_type)

