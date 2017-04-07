import numpy as np			# Array manipulation
import cv2				# Image manipulation
import xml.etree.ElementTree as ET	# To read XML docs

file_listdir = "../data/VOC2012/ImageSets/Main/"
img_listdir = "../data/VOC2012/JPEGImages/"
xml_listdir = "../data/VOC2012/Annotations/"
dest_dir = "./output/"

NUM_CHANS = 3
VECT_DIMS = 3

class vec:
	def populate_array(self, sample_list, MAX_X, MAX_Y, obj_type):
		# Create the output arrays for Keras
		x = np.zeros([len(sample_list), MAX_Y, MAX_X, NUM_CHANS], dtype=np.uint8)
		y = np.zeros([len(sample_list), VECT_DIMS], dtype=np.float32)
		# Populate those arrays	
		single_obj_count = 0
		for idx,sample in enumerate(sample_list):
			# Load the XML doc related to the image
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
				if(obj.find('name').text == str(obj_type)):
					obj_type_count += 1
					bnd = obj.find('bndbox')
					x_min = int(bnd.find('xmin').text) + (MAX_X/2-width/2)
					y_min = int(bnd.find('ymin').text) + (MAX_Y/2-height/2)
					x_max = int(bnd.find('xmax').text) + (MAX_X/2-width/2)
					y_max = int(bnd.find('ymax').text) + (MAX_Y/2-height/2)
					cent_x = (x_max-x_min)/2 + x_min
					cent_y = (y_max-y_min)/2 + y_min
					vec_x = (img_cent_x - cent_x)/float(MAX_X)
					vec_y = (img_cent_y - cent_y)/float(MAX_Y)
					vec_z = (x_max-x_min)*(y_max-y_min)/(float(width*height))
				else:
					vec_x = 0
					vec_y = 0
					vec_z = 0	

			if(obj_type_count <= 1):
				# Load the image 
				img = cv2.imread(img_listdir+sample+".jpg")
				height, width, channels = img.shape
				#print x_train.shape
				x[single_obj_count, (MAX_Y/2-height/2):(MAX_Y/2-height/2+height), (MAX_X/2-width/2):(MAX_X/2-width/2+width), :] = img
				temp_img = x[single_obj_count, :, :, :]
				#cv2.imshow('Centered Image', temp_img)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows()
				# Draw the vector on the image
				#lin_s = int(vec_z*40+1)
				#print lin_s
				#vec_img = cv2.line(x[single_obj_count, :, :, :], (cent_x, cent_y), (img_cent_x, img_cent_y), (200,100,50), lin_s)	
				#vec_img = cv2.circle(vec_img, (img_cent_x, img_cent_y),  lin_s, (0,255,0), 2)	
				#cv2.imshow('Centered Image', vec_img)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows()
				#cv2.imwrite(dest_dir+str(idx)+'_real.jpg', vec_img)
				# Make the labels
				y[single_obj_count, 0] = vec_x
				y[single_obj_count, 1] = vec_y
				y[single_obj_count, 2] = vec_z
				single_obj_count += 1

		# Trim the output array (delete empty entries)
		mask = np.ones(len(sample_list), dtype=bool)
		mask[range(single_obj_count, len(sample_list))] = False
		x = x[mask, :, :, :]
		y = y[mask, :]

		return (x, y)
	
	# Train and validation are only filled with birds			
	def load_pure(self, obj_type="bird"):
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
			if(spl[1] == '1'):
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

		return self.populate_array(train_list, MAX_X, MAX_Y, obj_type), self.populate_array(trainval_list, MAX_X, MAX_Y, obj_type), self.populate_array(val_list, MAX_X, MAX_Y, obj_type)
	
	# Training array has samples with birds and without birds present
	def load_messy(self, obj_type="bird"):
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

		return self.populate_array(train_list, MAX_X, MAX_Y, obj_type), self.populate_array(trainval_list, MAX_X, MAX_Y, obj_type), self.populate_array(val_list, MAX_X, MAX_Y, obj_type)
	

