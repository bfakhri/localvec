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
