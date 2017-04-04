# Converts the bounding boxes to vectors from the center

# To read XML docs
import xml.etree.ElementTree as ET
# To get filenames in directory
import glob

og_files = glob.glob("./Annotations/*.xml")

total_objects = 0
good_objects = 0
good_files = 0

for f in og_files:
	#print f
	tree = ET.parse(f)
	root = tree.getroot()
	# Get size of image
	s = root.find('size')
	width = int(s.find('width').text)
	height = int(s.find('height').text)
	img_cent_x = width/2
	img_cent_y = height/2
	good_file = True
	for obj in root.findall('object'):
		total_objects = total_objects + 1
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
		#print(str(vec_x) + " " + str(vec_y) + " " + str(vec_z))
		if(x_min == 0 or x_max == width or y_min == 0 or y_max == height):
			print("Object hits bounds in file: " + f)
			good_file = False
		else:
			good_objects = good_objects + 1
	if(good_file == True):
		good_files = good_files + 1

print("Good Objects = " + str(good_objects) + "/" + str(total_objects) + " = " + str(100*good_objects/float(total_objects)) + "%")
print("Good Files = " + str(good_files) + "/" + str(len(og_files)))
		
