import numpy as np

file_listdir = "./VOC2012/ImageSets/Main/"
img_listdir = "./VOC2012/JPEGImages/"

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

	# Create the x_train and y_train vectors
	

cor()
