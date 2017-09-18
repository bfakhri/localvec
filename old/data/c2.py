# Converts the bounding boxes to vectors from the center

# To read XML docs
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree
import sys
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
		# If bbox hits a side, we don't zoom further
		if(x_min == 0 or x_max == width or y_min == 0 or y_max == height):
			print("Object hits bounds in file: " + f)
			vec_z = 0
		else:
			# Is this the best way to represent zoom? 
			vec_z = (x_max-x_min)*(y_max-y_min)/(float(width*height))
		print(str(vec_x) + " " + str(vec_y) + " " + str(vec_z))
		zoom_vec = Element('zoomvector')
		ET.SubElement(zoom_vec, str(vec_x))
		ET.SubElement(zoom_vec, str(vec_y))
		ET.SubElement(zoom_vec, str(vec_z))
		obj.append(zoom_vec)
	# Write out to file
	ElementTree(root).write(sys.stdout, encoding='utf-8')
		
