import numpy as mp
import cv2

'''
Much of this code comes from the tutorial found at:
http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

'''

'''
bb_iou takes in two bounding boxes (in the form [xmin, ymin, xmax, ymax])
and calculates their intersection over union.
'''
def bb_iou(boxA, boxB):
	#determine the coordinates of the intersection
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	#compute the area of intersection rectangle
	interArea = (xB - xA + 1)*(yB-yA +1)  #add one for offset, 

	#compute the area of the predicion and ground-truth rect
	boxA_area = (boxA[2] - boxA[0] +1) * (boxA[3] - boxA[1] + 1)
	boxB_area = (boxB[2] - boxB[0] +1) * (boxB[3] - boxB[1] + 1)

	iou = interArea/float(boxA_area + boxB_area -interArea) #subtract interarea because it's counted twice in the total area

	return iou