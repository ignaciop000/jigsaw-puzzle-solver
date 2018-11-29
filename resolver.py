##USAGE
##	python resolver.py --images pieces
import argparse
import imutils
import os
import sys
import cv2
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import scipy
import scipy.stats

'''
Function : cv2.cornerHarris(image,blocksize,ksize,k)
Parameters are as follows :
1. image : the source image in which we wish to find the corners (grayscale)
2. blocksize : size of the neighborhood in which we compare the gradient 
3. ksize : aperture parameter for the Sobel() Operator (used for finding Ix and Iy)
4. k : Harris detector free parameter (used in the calculation of R)
'''

def harris_corners(image):
	
	#Converting the image to grayscale
	gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
	#Conversion to float is a prerequisite for the algorithm
	gray_img = np.float32(gray_img)
	
	# 3 is the size of the neighborhood considered, aperture parameter = 3
	# k = 0.04 used to calculate the window score (R)
	corners_img = cv2.cornerHarris(gray_img,3,3,0.04)
	
	#Marking the corners in Green
	image[corners_img>0.001*corners_img.max()] = [0,255,0]
	xy = get_corners(corners_img,5, 0.2,100)
	return (image, xy)

'''
Function: cv2.goodFeaturesToTrack(image,maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
image  Input 8-bit or floating-point 32-bit, single-channel image.
maxCorners  You can specify the maximum no. of corners to be detected. (Strongest ones are returned if detected more than max.)
qualityLevel  Minimum accepted quality of image corners.
minDistance  Minimum possible Euclidean distance between the returned corners.
corners  Output vector of detected corners.
mask  Optional region of interest. 
blockSize  Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. 
useHarrisDetector  Set this to True if you want to use Harris Detector with this function.
k  Free parameter of the Harris detector.
'''

def shi_tomasi(image):

	#Converting to grayscale
	gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
	#Specifying maximum number of corners as 1000
	# 0.01 is the minimum quality level below which the corners are rejected
	# 10 is the minimum euclidean distance between two corners
	corners_img = cv2.goodFeaturesToTrack(gray_img,1000,0.01,10)
	
	corners_img = np.int0(corners_img)
	xy = []
	for corners in corners_img:      
		x,y = corners.ravel()
		xy.append((x,y))
		#Circling the corners in green
		#cv2.circle(image,(x,y),3,[0,0,255],-1)
	draw_points(image, xy)
	return (image,xy)

def draw_points(image, points, color = [0,0,255]):
	for x,y in points:
		cv2.circle(image,(x,y),3,color,-1)
	return image;

def draw_line_withCoef(image, a, b, c):
	"""
	ax + by + c = 0
	y = (-c - ax)/b
	"""
	if a == 0:
		print "if"
		#x = 0
		#y = -c/b
		x0 = 0
		y0 = int(-c/b)
		x1 = 1
		y1 = int(-c/b)
	elif b == 0:
		print "elif"
		#y = 0
		#x = -c/a
		x0 = int(-c/a)
		y0 = 0
		x1 = int(-c/a)
		y1 = 1
	else:
		x0 = 0		
		y0 = int( -c / b )
		x1 = 1
		y1 = int((-c-a)/b)
	print x0, y0, x1, y1
	lineThickness = 2
	color = (0, 255,0)
	cv2.line(image, (x0, y0), (x1, y1), color, lineThickness)
	return image

def draw_contour(image, contour, index):
	cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
	M = cv2.moments(contour)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	# draw the countour number on the image
	cv2.putText(image, "#{}".format(index + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

	# return the image with the contour number drawn on it
	return image

def sort_contours(cnts, imageShape):

	# construct the list of bounding boxes and sort them from top to bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	max_width = max(boundingBoxes, key=lambda r: r[0] + r[2])[0]
	max_height = max(boundingBoxes, key=lambda r: r[3])[3]
	nearest = max_height * 1.4
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:(int(nearest * round(float(b[1][1])/nearest)) * max_width + b[1][0])))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def segment_piece(image, bin_threshold=128):
	"""
	Apply segmentation of the image by simple binarization
	"""
	return cv2.threshold(image, bin_threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

def compute_minmax_xy(thresh):
	"""
	Given the thresholded image, compute the minimum and maximum x and y 
	coordinates of the segmented puzzle piece.
	"""
	idx_shape = np.where(thresh == 0)
	return [np.array([coords.min(), coords.max()]) for coords in idx_shape]


def extract_piece(thresh):

	# Here we build a square image centered on the blob (piece of the puzzle).
	# The image is constructed large enough to allow for piece rotations. 
	
	minmax_y, minmax_x = compute_minmax_xy(thresh)

	ly, lx = minmax_y[1] - minmax_y[0], minmax_x[1] - minmax_x[0]
	size = int(max(ly, lx) * np.sqrt(2))

	x_extract = thresh[minmax_y[0]:minmax_y[1] + 1, minmax_x[0]:minmax_x[1] + 1]
	ly, lx = x_extract.shape

	xeh, xew = x_extract.shape
	x_copy = np.full((size, size), 255, dtype='uint8')
	sy, sx = size // 2 - ly // 2, size // 2 - lx // 2

	x_copy[sy: sy + ly, sx: sx + lx] = x_extract
	thresh = x_copy
	thresh = 255 - thresh
	return thresh

def get_corners(dst, neighborhood_size=5, score_threshold=0.3, minmax_threshold=100):
	
	"""
	Given the input Harris image (where in each pixel the Harris function is computed),
	extract discrete corners
	"""
	data = dst.copy()
	data[data < score_threshold*dst.max()] = 0.

	data_max = filters.maximum_filter(data, neighborhood_size)
	maxima = (data == data_max)
	data_min = filters.minimum_filter(data, neighborhood_size)
	diff = ((data_max - data_min) > minmax_threshold)
	maxima[diff == 0] = 0

	labeled, num_objects = ndimage.label(maxima)
	slices = ndimage.find_objects(labeled)
	yx = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
	return yx[:, ::-1]

def compute_angles(xy):
	N = len(xy)
	angles = np.zeros((N, N))

	for i in range(N):
		for j in range(i + 1, N):

			point_i, point_j = xy[i], xy[j]
			if point_i[0] == point_j[0]:
				angle = 90
			else:
				angle = np.arctan2(point_j[1] - point_i[1], point_j[0] - point_i[0]) * 180 / np.pi

			angles[i, j] = angle
			angles[j, i] = angle

	return angles

def search_for_possible_rectangle(xy, distances, angles, idx, prev_points, perp_angle_thresh, verbose, possible_rectangles):
	curr_point = xy[idx]
	depth = len(prev_points)

	if depth == 0:
		right_points_idx = np.nonzero(np.logical_and(xy[:, 0] > curr_point[0], distances[idx] > 0))[0]		
		if verbose >= 2:
			print 'point', idx, curr_point
		for right_point_idx in right_points_idx:
			search_for_possible_rectangle(xy, distances, angles, right_point_idx, [idx], perp_angle_thresh, verbose, possible_rectangles)

		if verbose >= 2:
			print
			
		return


	last_angle = angles[idx, prev_points[-1]]
	perp_angle = last_angle - 90
	if perp_angle < 0:
		perp_angle += 180

	if depth in (1, 2):

		if verbose >= 2:
			print '\t' * depth, 'point', idx, '- last angle', last_angle, '- perp angle', perp_angle

		diff0 = np.abs(angles[idx] - perp_angle) <= perp_angle_thresh
		diff180_0 = np.abs(angles[idx] - (perp_angle + 180)) <= perp_angle_thresh
		diff180_1 = np.abs(angles[idx] - (perp_angle - 180)) <= perp_angle_thresh
		all_diffs = np.logical_or(diff0, np.logical_or(diff180_0, diff180_1))
		
		diff_to_explore = np.nonzero(np.logical_and(all_diffs, distances[idx] > 0))[0]

		#if verbose >= 2:
		#	print '\t' * depth, 'diff0:', np.nonzero(diff0)[0], 'diff180:', np.nonzero(diff180)[0], 'diff_to_explore:', diff_to_explore

		for dte_idx in diff_to_explore:
			if dte_idx not in prev_points: # unlickly to happen but just to be certain
				next_points = prev_points[::]
				next_points.append(idx)

				search_for_possible_rectangle(xy, distances, angles, dte_idx, next_points, perp_angle_thresh, verbose, possible_rectangles)
			
	if depth == 3:
		angle41 = angles[idx, prev_points[0]]

		diff0 = np.abs(angle41 - perp_angle) <= perp_angle_thresh
		diff180_0 = np.abs(angle41 - (perp_angle + 180)) <= perp_angle_thresh
		diff180_1 = np.abs(angle41 - (perp_angle - 180)) <= perp_angle_thresh
		dist = distances[idx, prev_points[0]] > 0

		if dist and (diff0 or diff180_0 or diff180_1):
			rect_points = prev_points[::]
			rect_points.append(idx)
			
			if verbose == 2:
				print 'We have a rectangle:', rect_points

			already_present = False
			for possible_rectangle in possible_rectangles:
				if set(possible_rectangle) == set(rect_points):
					already_present = True
					break

			if not already_present:
				possible_rectangles.append(rect_points)

def PolyArea(x,y):
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def get_best_fitting_rect_coords(xy, d_threshold=30, perp_angle_thresh=20, verbose=0):

	"""
	Since we expect the 4 puzzle corners to be the corners of a rectangle, here we take
	all detected Harris corners and we find the best corresponding rectangle.
	We perform a recursive search with max depth = 2:
	- At depth 0 we take one of the input point as the first corner of the rectangle
	- At depth 1 we select another input point (with distance from the first point greater
		then d_threshold) as the second point
	- At depth 2 and 3 we take the other points. However, the lines 01-12 and 12-23 should be
		as perpendicular as possible. If the angle formed by these lines is too much far from the
		right angle, we discard the choice.
	- At depth 3, if a valid candidate (4 points that form an almost perpendicular rectangle) is found,
		we add it to the list of candidates.
		
	Given a list of candidate rectangles, we then select the best one by taking the candidate that maximizes
	the function: area * Gaussian(rectangularness)
	- area: it is the area of the candidate shape. We expect that the puzzle corners will form the maximum area
	- rectangularness: it is the mse of the candidate shape's angles compared to a 90 degree angles. The smaller
						this value, the most the shape is similar toa rectangle.
	"""
	N = len(xy)

	distances = scipy.spatial.distance.cdist(xy, xy)
	distances[distances < d_threshold] = 0

	angles = compute_angles(xy)
	possible_rectangles = []
	"""
	if verbose >= 2:
		print 'Coords'
		print xy
		print
		print 'Distances'
		print distances
		print
		print 'Angles'
		print angles
		print
	"""
	for i in range(N):
		search_for_possible_rectangle(xy, distances, angles, i, [], perp_angle_thresh, verbose, possible_rectangles)                 
					
	if len(possible_rectangles) == 0:
		return None

	areas = []
	rectangularness = []
	diff_angles = []

	for r in possible_rectangles:
		points = xy[r]
		areas.append(PolyArea(points[:, 0], points[:, 1]))

		mse = 0
		da = []
		for i1, i2, i3 in [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]:
			diff_angle = abs(angles[r[i1], r[i2]] - angles[r[i2], r[i3]])
			da.append(abs(diff_angle - 90))
			mse += (diff_angle - 90) ** 2

		diff_angles.append(da)
		rectangularness.append(mse)


	areas = np.array(areas)
	rectangularness = np.array(rectangularness)

	scores = areas * scipy.stats.norm(0, 150).pdf(rectangularness)
	best_fitting_idxs = possible_rectangles[np.argmax(scores)]
	return xy[best_fitting_idxs]

def rotate(image, degrees):
	"""
	Rotate an image by the amount specifiedi in degrees
	"""
	if len(image.shape) == 3:
		rows,cols, _ = image.shape
	else:
		rows, cols = image.shape
		
	M = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
	
	return cv2.warpAffine(image,M,(cols,rows)), M

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)), M	

def compute_barycentre(thresh, value=0):
	"""
	Given the segmented puzzle piece, compute its barycentre.
	"""
	idx_shape = np.where(thresh == value)
	return [int(np.round(coords.mean())) for coords in idx_shape]

def corner_detection(edges, intersections, (xb, yb), rect_size=50, show=False):

	# Find corners by taking the highest distant point from a 45 degrees inclined line
	# inside a squared ROI centerd on the previously found intersection point.
	# Inclination of the line depends on which corner we are looking for, and is
	# computed based on the position of the barycenter of the piece.

	corners = []

	for idx, intersection in enumerate(intersections):
			
		xi, yi = intersection

		m = -1 if (yb - yi)*(xb - xi) > 0 else 1
		y0 = 0 if yb < yi else 2*rect_size
		x0 = 0 if xb < xi else 2*rect_size

		a, b, c = m, -1, -m*x0 + y0

		rect = edges[yi - rect_size: yi + rect_size, xi - rect_size: xi + rect_size].copy()

		edge_idx = np.nonzero(rect)
		if len(edge_idx[0]) > 0:
			distances = [(a*edge_x + b*edge_y + c)**2 for edge_y, edge_x in zip(*edge_idx)]
			corner_idx = np.argmax(distances)

			rect_corner = np.array((edge_idx[1][corner_idx], edge_idx[0][corner_idx]))
			offset_corner = rect_corner - rect_size
			real_corner = intersection + offset_corner

			corners.append(real_corner)
		else:
			# If the window is completely black I can make no assumption: I keep the same corner
			corners.append(intersection)

		if show:
			plt.subplot(220 + idx + 1)
			cv2.circle(rect, tuple(rect_corner), 5, 128)
			
			plt.title("{0} | {1}".format(intersection, (x0, y0)))
			plt.imshow(rect)
	
	if show:
		plt.show()
		
	return corners

def order_corners(corners):
	corners.sort(key=lambda k: k[0] + k[1])
	antidiag_corners = sorted(corners[1:3], key=lambda k: k[1])
	corners[1:3] = antidiag_corners
	return corners

def compute_line_params(corners):
	return [get_line_through_points(corners[i1], corners[i2]) for i1, i2 in _corner_indexes]


def shape_classification(edges, line_params, d_threshold=500, n_hs=10):
	
	# First part: we take all edge points and classify them only if their distance to one of the 4 piece
	# lines is smaller than a certain threshold. If that happens, we can be certain that the point belongs
	# to that side of the piece. If each one of the four distances is higher than the threshold, the point
	# will be classified during the second phase.

	y_nonzero, x_nonzero = np.nonzero(edges)
	distances = []

	class_image = np.zeros(edges.shape, dtype='uint8')
	non_classified_points = []

	for x_edge, y_edge in zip(x_nonzero, y_nonzero):
		d = [distance_point_line_squared(line_param, (x_edge, y_edge)) for line_param in line_params]
		if np.min(d) < d_threshold:
			class_image[y_edge, x_edge] = np.argmin(d) + 1
		else:
			non_classified_points.append((x_edge, y_edge))

	non_classified_points = np.array(non_classified_points)

	# Second part: hysteresis classification
	# Edge points that have not been classified because they are too far from all lines
	# will be classified based on their neighborood: if the neighborhood of a point contains
	# an already classified point, it will be classified with the same class.
	# It's very unlikely that the neighborhood of a non classified point will contain two different
	# classes, so we just take the first non-zero value that we find inside the neighborhood
	# The process is repeated and at each iteration the newly classified points are removed from the ones
	# that still need to be classified. The process is interrupted when no new point has been classified
	# or when a maximum number of iterations has been reached (in case of a noisy points that has no neighbours).
	
	
	map_iteration = 0
	max_map_iterations = 50

	while map_iteration < max_map_iterations:

		map_iteration += 1
		classified_points_at_current_iteration = []

		for idx, (x_edge, y_edge) in enumerate(non_classified_points):

			neighborhood = class_image[y_edge - n_hs: y_edge + n_hs + 1, x_edge - n_hs: x_edge + n_hs + 1]
			n_mapped = np.nonzero(neighborhood)
			if len(n_mapped[0]) > 0:
				ny, nx = n_mapped[0][0] - n_hs, n_mapped[1][0] - n_hs
				class_image[y_edge, x_edge] = class_image[y_edge + ny, x_edge + nx]
				classified_points_at_current_iteration.append(idx)

		if len(non_classified_points) > 0:
			non_classified_points = np.delete(non_classified_points, classified_points_at_current_iteration, axis=0)
		else:
			break
			
	return class_image


def get_line_through_points(p0, p1):
	"""
	Given two points p0 (x0, y0) and p1 (x1, y1),
	compute the coefficients (a, b, c) of the line 
	that passes through both points.
	"""
	x0, y0 = p0
	x1, y1 = p1
	
	return y1 - y0, x0 - x1, x1*y0 - x0*y1

def distance_point_line_squared((a, b, c), (x0, y0)):
	"""
	Computes the squared distance of a 2D point (x0, y0) from a line ax + by + c = 0
	"""
	return (a*x0 + b*y0 + c)**2 / (a**2 + b**2)


def distance_point_line_signed((a, b, c), (x0, y0)):
	"""
	Computes the signed distance of a 2D point (x0, y0) from a line ax + by + c = 0
	"""
	return (a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)


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

	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	
	thresh =cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

	# noise removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
	edged = cv2.Canny(opening, 30, 200)
	#cv2.imshow("gray", imutils.resize( gray, height = 1000))
	#cv2.imshow("blur", imutils.resize( blur, height = 1000))
	#cv2.imshow("thresh", imutils.resize( thresh, height = 1000))
	#cv2.imshow("opening", imutils.resize( opening, height = 1000))
	#cv2.imshow("edged", imutils.resize( edged, height = 1000))

	(_, cnts, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:40]
	orig = image.copy()
	# sort the contours according to the provided method
	(cnts, boundingBoxes) = sort_contours(cnts, image.shape)
	#for x, y, w, h in boundingBoxes:
	#	print "{:4} {:4} {:4} {:4}".format(x, y, w, h) 

	for (i, c) in enumerate(cnts):
		try:		
			orig = draw_contour(orig, c, i)	
			x,y,w,h = cv2.boundingRect(c)
			cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)
			blob = opening[y:y+h+3 ,x:x+w+3]
			blob_color=cv2.merge((blob,blob,blob))
			blob_gray = cv2.cvtColor(blob_color, cv2.COLOR_BGR2GRAY)
			#shiTomasi,xy = shi_tomasi(blob_color.copy())
			harris, xy  = harris_corners(blob_color.copy())							
			_corner_indexes = [(0, 1), (1, 3), (3, 2), (0, 2)]
			"""
			img = np.float32(opening_blob)
			harris = cv2.cornerHarris(img, 2, 3, 0.04)
			harris = harris * gray_blob			
			xy = np.round(xy).astype(np.int)
			"""
			xy = np.round(xy).astype(np.int)
			harris_processed = draw_points(blob_color.copy(),xy)
			
			if len(xy) < 4:
				raise RuntimeError('Not enough corners')

			intersections = get_best_fitting_rect_coords(xy, perp_angle_thresh=30)
			harris_intersections = draw_points(blob_color.copy(),intersections)
			#print "intersections", intersections

			if intersections is None:
				raise RuntimeError('No rectangle found')	
			if intersections[1, 0] == intersections[0, 0]:
				rotation_angle = 90
			else:
				rotation_angle = np.arctan2(intersections[1, 1] - intersections[0, 1], intersections[1, 0] - intersections[0, 0]) * 180 / np.pi
			
			edges = blob_gray - cv2.erode(blob_gray, np.ones((3, 3)))
			#plt.figure(figsize=(6, 6))
			#plt.title("{0} - {1}".format(filename, label))
			#plt.imshow(gray_blob, cmap='gray')
			#print xy
			#plt.scatter(xy[:, 0], xy[:, 1], color='red')
			#plt.scatter(intersections[:, 0], intersections[:, 1], color='blue')
			#plt.colorbar()
			#plt.show()
			# Rotate all images
			edges, M = rotate_bound(edges, rotation_angle)			
			edges_color=cv2.merge((edges,edges,edges))
			# Rotate intersection points
			intersections = np.array(np.round([M.dot((point[0], point[1], 1)) for point in intersections])).astype(np.int)
			rotate_intersections = draw_points(edges.copy(),intersections)
		
			yb, xb = compute_barycentre(edges)
		
			corners = corner_detection(edges, intersections, (xb, yb), 5, show=False)
			corners = order_corners(corners)
			edge_corners = draw_points(edges_color.copy(),corners)
			#blob_corners = draw_points(edges.copy(),corners)
			line_params = compute_line_params(corners)
			for a,b,c in line_params
				edge_corners = draw_line_withCoef(edge_corners, a,b,c)
			class_image = shape_classification(edges, line_params, 100, 5)
			print class_image
	
			cv2.imshow("blob", blob)
			#cv2.imshow("gray_blob", gray_blob)
			#cv2.imshow("thresh_blob", thresh_blob)
			#cv2.imshow("thresh_blob", opening_blob)
			#cv2.imshow("harris", harris)
			#cv2.imshow("shi_tomasi", shiTomasi)
			#cv2.imshow("harris_processed", harris_processed)
			#cv2.imshow("harris_intersections", harris_intersections)			
			#cv2.imshow("edges", edges)
			#cv2.imshow("rotate_intersections", rotate_intersections)
			cv2.imshow("edge_corners", edge_corners)
			
			cv2.imshow("class_image", class_image)
			cv2.waitKey(0)
		except Exception as e:
			print e
		#finally:
		#	cv2.waitKey(0)

		

	#cv2.imshow("Processed", imutils.resize( orig, height = 1000))
	#cv2.waitKey(0)
	cv2.destroyAllWindows()