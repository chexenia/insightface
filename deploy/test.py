import face_embedding
import argparse
import cv2
import numpy as np
import reco2.common as cmn
import os
import time

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r34-amf/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--db', default='.', help = 'Path to input db.' )
parser.add_argument('--outpath', default='.', help='Path to store output.' )
parser.add_argument('--viz', default=False, help='if you need to vizualize landmarks')

args = parser.parse_args()

scans = cmn.enumerateFiles(args.db, ['.png'])

start = time.time()

model = face_embedding.FaceModel(args)

for scan in scans:
	print(scan)
	img = cv2.imread(scan)

	outpath = os.path.splitext(scan.replace(args.db, args.outpath))[0]
	outfit = outpath + '.insightface'

	res = model.get_feature(img)
	if res is not None:
		f1, bboxes, points = res
		if not os.path.exists(os.path.dirname(outfit)):
			os.makedirs(os.path.dirname(outfit))

		np.savetxt(outfit, f1)

		if args.viz:
			cv2.rectangle(img, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), (0, 0, 255))
			for pt in points:
				cv2.circle(img, (pt[0], pt[1]), 2, (0, 0, 255))
			cv2.imwrite(outpath + '_mtcnn.png', img)	
	else:
		print('..failed')
print ("Time elapsed: ", time.time() - start)