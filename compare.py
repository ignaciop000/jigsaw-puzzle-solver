import os
import numpy as np
import cv2

files = os.listdir("curves")
curves = []
for filename in files:
	curve = np.load(os.path.join("curves", filename))
	curves.append((filename, curve))
for filename1, curve1 in curves:
	for filename2, curve2 in curves:
		ret = cv2.matchShapes(curve1,curve2,1,0.0)
		if (ret < 10 and filename1 != filename2):
			print "Pieza Canditata", filename1, filename2, ret
	