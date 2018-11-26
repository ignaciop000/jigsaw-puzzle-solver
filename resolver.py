##USAGE
##	python resolver.py --images images
import argparse
import os
import sys
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the input images")

args = vars(ap.parse_args())

files = os.listdir(args["images"])
for filename in files:
	print os.path.join('images', filename)
	img = cv2.imread(os.path.join('images', filename))