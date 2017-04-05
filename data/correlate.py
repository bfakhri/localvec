import numpy as np			# Array manipulation
import cv2				# Image manipulation
import xml.etree.ElementTree as ET	# To read XML docs

file_listdir = "./VOC2012/ImageSets/Main/"
img_listdir = "./VOC2012/JPEGImages/"
xml_listdir = "./VOC2012/Annotations/"

NUM_CHANS = 3
VECT_DIMS = 3

def cor(obj_type="bird"):
	# Get handle to list of files 
	trn_str = file_listdir+obj_type+"_"+"train.txt"
	val_str = file_listdir+obj_type+"_"+"val.txt"
	print("Opening dataset with type: "+obj_type)
	print("Opening "+trn_str)
	print("Opening "+val_str)
	train_file_hdl = open(trn_str, "r")
	val_file_hdl = open(val_str, "r")

	train_list = []
	val_list = []
	# Put into vectors
	for line in train_file_hdl:
		spl = line.split()
		if(spl[1] == '1'):
			print("Train Saved: " + line)
			train_list.append(spl[0])
	for line in val_file_hdl:
		spl = line.split()
		if(spl[1] == '1'):
			print("Val Saved: " + line)
			val_list.append(spl[0])

	print("There are : " + str(len(train_list)) + " training samples")
	print("There are : " + str(len(val_list)) + " validation samples")

	# A little silly - but we have to find the max resolution of all images for both dims
	MAX_X = 0
	MAX_Y = 0
	for sample in set(val_list).union(train_list):
		img = cv2.imread(img_listdir+sample+".jpg")
		height, width, channels = img.shape
		if(height > MAX_Y):
			MAX_Y = height
		if(width > MAX_X):
			MAX_X = width

	print("Max Image Dimensions: " + str(MAX_X) + "x" + str(MAX_Y))	

	# Create the output arrays for Keras
	x_train = np.zeros([len(train_list), MAX_Y, MAX_X, NUM_CHANS], dtype=np.uint8)
	y_train = np.zeros([len(train_list), VECT_DIMS], dtype=np.float32)
	x_test = np.zeros([len(val_list), MAX_Y, MAX_X, NUM_CHANS], dtype=np.uint8)
	y_test = np.zeros([len(val_list), VECT_DIMS], dtype=np.float32)
	
	# Populate those arrays	
	single_obj_count = 0
	for sample in train_list:
		# Load the XML doc related to the image
		print("Opening: " +xml_listdir+sample+".xml") 
		tree = ET.parse(xml_listdir+sample+".xml")
		root = tree.getroot()
		# Get size of image
		s = root.find('size')
		width = int(s.find('width').text)
		height = int(s.find('height').text)
		img_cent_x = MAX_X/2
		img_cent_y = MAX_Y/2
		obj_type_count = 0
		for obj in root.findall('object'):
			if(obj.find('name').text == obj_type):
				obj_type_count += 1
				bnd = obj.find('bndbox')
				x_min = int(bnd.find('xmin').text) + (MAX_X/2-width/2)
				y_min = int(bnd.find('ymin').text) + (MAX_Y/2-height/2)
				x_max = int(bnd.find('xmax').text) + (MAX_X/2-width/2)
				y_max = int(bnd.find('ymax').text) + (MAX_Y/2-height/2)
				cent_x = (x_max-x_min)/2 + x_min
				cent_y = (y_max-y_min)/2 + y_min
				vec_x = img_cent_x - cent_x
				vec_y = img_cent_y - cent_y
				# Is this the best way to represent zoom? 
				vec_z = (x_max-x_min)*(y_max-y_min)/(float(width*height))
				print(str(vec_x) + " " + str(vec_y) + " " + str(vec_z))
			else:
				vec_x = 0
				vec_y = 0
				vec_z = -1	
				print(str(vec_x) + " " + str(vec_y) + " " + str(vec_z))

		if(obj_type_count == 1):
			# Load the image 
			img = cv2.imread(img_listdir+sample+".jpg")
			height, width, channels = img.shape
			x_train[single_obj_count, (MAX_Y/2-height/2):(MAX_Y/2-height/2+height), (MAX_X/2-width/2):(MAX_X/2-width/2+width), :] = img
			temp_img = x_train[single_obj_count, :, :, :]
			#cv2.imshow('Centered Image', temp_img)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()
			# Draw the vector on the image
			vec_img = cv2.line(x_train[single_obj_count, :, :, :], (cent_x, cent_y), (img_cent_x, img_cent_y), (255,0,0),5)	
			cv2.imshow('Centered Image', vec_img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			# Make the labels
			y_train[single_obj_count, 0] = vec_x
			y_train[single_obj_count, 1] = vec_y
			y_train[single_obj_count, 2] = vec_z
			single_obj_count += 1

				
				
		

		

cor()