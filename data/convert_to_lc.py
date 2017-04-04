# Converts the bounding boxes to vectors from the center

# To read XML docs
import xml.etree.ElementTree as ET
# To get filenames in directory
import glob

og_files = glob.glob("./Annotations/*.xml")

for f in og_files:
	print f
	tree = ET.parse(f)
	root = tree.getroot()
	# Get size of image
	s = root.find('size')
	width = int(s.find('width').text)
	height = int(s.find('height').text)
	img_cent_x = width/2
	img_cent_y = height/2
	for obj in root.findall('object'):
		bnd = obj.find('bndbox')
		x_min = int(bnd.find('xmin').text)
		y_min = int(bnd.find('ymin').text)
		x_max = int(bnd.find('xmax').text)
		y_max = int(bnd.find('ymax').text)
		cent_x = (x_max-x_min)/2 + x_min
		cent_y = (y_max-y_min)/2 + y_min
		vec_x = img_cent_x - cent_x
		vec_y = img_cent_y - cent_y
		# Is this the best way to represent zoom? 
		vec_z = (x_max-x_min)*(y_max-y_min)/(float(width*height))
		print(str(vec_x) + " " + str(vec_y) + " " + str(vec_z))
		
