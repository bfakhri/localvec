def bbox_IoU(bb_pred, bb_gt):
	# Calculate the area of each box
	prediction_area = (bb_pred[0] - bb_pred[2])*(bb_pred[1] - bb_pred[3])
	groundtruth_area = (bb_gt[0] - bb_gt[2])*(bb_gt[1] - bb_gt[3])
	
	# Calculate the intersection rectangle bounds
	topleft_x = max(bb_pred[0], bb_gt[0])
	bottomright_x = min(bb_pred[2], bb_gt[2])
	topleft_y = max(bb_pred[1], bb_gt[1])
	bottomright_y = min(bb_pred[3], bb_gt[3])
	# Calculate its area
	inter_area = (bottomright_x - topleft_x)*(bottomright_y - topleft_y)

	# Calculate IoU
	IoU = inter_area/(prediction_area + groundtruth_area - inter_area)
	if(IoU < 0):
		return 0
	else:
		return IoU
