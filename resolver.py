##USAGE
##	python resolver.py --images images
##	python resolver.py --images pieces
import argparse
import imutils
import os
import sys
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the input images")

args = vars(ap.parse_args())

files = os.listdir(args["images"])
for filename in files:
	#Leer imagen, fondo oscuro, Hoja blanca y las piezas
	print os.path.join(args["images"], filename)
	image = cv2.imread(os.path.join(args["images"], filename))
	#Obtener contorno de la hoja. 
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)
	#Normalizar hoja
	#Divir piezas
	
	#Apply the four point transform to obtain a top-down
