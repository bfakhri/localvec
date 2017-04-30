'''
-----------------------------------------------------------------------------
LICENSE
-----------------------------------------------------------------------------
The MIT License

Copyright (c) 2017 - Bijan Fakhri and Meredith Moore

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
This file calculates the Intersection over Union measure of two bounding 
boxes. The measure is used to compare localization algorithms. 

Written by Bijan Fakhri and Meredith Moore
-----------------------------------------------------------------------------
'''

def bbox_IoU(bb_pred, bb_gt):
	# Calculate the area of each box
	prediction_area = (bb_pred[0] - bb_pred[2])*(bb_pred[1] - bb_pred[3])
	print("pa" + str(prediction_area))
	groundtruth_area = (bb_gt[0] - bb_gt[2])*(bb_gt[1] - bb_gt[3])
	print("gta" + str(groundtruth_area))
	
	# Calculate the intersection rectangle bounds
	topleft_x = max(bb_pred[0], bb_gt[0])
	bottomright_x = min(bb_pred[2], bb_gt[2])
	topleft_y = max(bb_pred[1], bb_gt[1])
	bottomright_y = min(bb_pred[3], bb_gt[3])
	# Calculate its area
	inter_area = (bottomright_x - topleft_x)*(bottomright_y - topleft_y)
	print("ia" + str(inter_area))

	# Calculate IoU
	IoU = float(inter_area)/float(prediction_area + groundtruth_area - inter_area)
	print("IoU: " + str(IoU))
	if(IoU < 0):
		return 0
	else:
		return IoU
