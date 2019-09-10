#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
import math

import load_obj
import frame_obj


'''
xc=310.1 # optical centres
yc=230.1
fx=730.2 # focal length
fy=729.4
k1=0.131 # radial distortion coefficients
k2=-0.336
p1=-0.00228 # tangential distortion coefficients
p2=-0.00546
'''

xc=310.1154139938381 # optical centres
yc=230.51439408952078
fx=730.1622648656321 # focal length
fy=729.4304579639811
k1=0.13062577479871762 # radial distortion coefficients
k2=-0.33648983794113113
p1=-0.0022840799601023302 # tangential distortion coefficients
p2=-0.0054561685500777507

MIN_MATCH_COUNT = 10 # homography need at least 4 good points

distortion_coefficients = np.float32([k1, k2, p1, p2])
camera_matrix = np.float32([[fx, 0.0, xc], [0.0, fy, yc], [0.0, 0.0, 1.0]])

smoothing = True


class Program:

	def __init__(self, alg, test_param):
		self.two_points = []
		self.selecting = False
		self.selected_frame = None 
		self.alg = alg 
		self.test_param = test_param


	def select_frame(self, event, x, y, flags, test_param):

		if event == cv2.EVENT_LBUTTONDOWN:

			self.two_points = [(x, y)]
			self.selecting = True
	 
		elif event == cv2.EVENT_LBUTTONUP:

			self.two_points.append((x, y))
			self.selecting = False



	def render_test(self, img, imgpts):
		imgpts = np.int32(imgpts).reshape(-1,2)

		img = cv2.drawContours(img, [imgpts[:4]],-1,(250,0,0),-3) # red floor

		return img


	def render(self, img, obj, projection, points_memory):

		vertices = obj.vertices
		scale_matrix = np.eye(3) * 40 # scale of object kubek

		h,w = self.selected_frame.image.shape 
		imgpts = 0

		if smoothing == True:
			if len(points_memory) > 10:
				points_memory.insert(0,projection)
				projection = sum(points_memory[:-9]) / 9
				points_memory.pop()

				#memor.append(projection)
			else:
				points_memory.insert(0,projection)
				#memor.append(projection)
			# 1 2 3 / 3 1 2 		


		for face in obj.faces:
			face_vertices = face
			points = np.array([vertices[vertex - 1] for vertex in face_vertices])
			points = np.dot(points, scale_matrix)
			
			# (w, h) / 2 to render mug in center
			points = np.array([[p[0] + w/2 , p[1] + h/2 , p[2]] for p in points])
			dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)

			imgpts = np.int32(dst)


			cv2.fillConvexPoly(img, imgpts, (84, 84, 84))
			#cv2.fillPoly(img, imgpts, (2, 2, 200))

		return (img, points_memory)

	

	def main(self):
		points_memory = []


		cap = cv2.VideoCapture(0)

		while True:
 
			ret, frame = cap.read()

			currentFrame = frame.copy()
			cv2.namedWindow("choose marker")
			cv2.setMouseCallback("choose marker", self.select_frame)
			
			cv2.imshow('choose marker',frame)


			if len(self.two_points) == 2:
				cropImage = currentFrame[self.two_points[0][1]:self.two_points[1][1], self.two_points[0][0]:self.two_points[1][0]]
				cv2.rectangle(currentFrame, self.two_points[0], self.two_points[1], (200, 200, 0), 2)

				self.selected_frame = frame_obj.Marker_frame(cropImage, self.alg)

				cv2.imshow('choose marker',currentFrame)
				cv2.waitKey(1000)
				cv2.destroyWindow('choose marker')
				break

			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break


		cv2.waitKey(100)
		matcher = Matcher(self.selected_frame, self.alg, distortion_coefficients, camera_matrix, test_param)
		
		
		while True:

			ret, frame = cap.read()
			currentFrame = frame.copy()

			cv2.namedWindow('webcam')
			cv2.imshow('webcam', currentFrame)
			
			matcher.setFrame(currentFrame)
			

			result = matcher.getCorrespondence()
			if result:
				(src, dst, corners, MH) = result
			else:
				print('Not enough points matched!')
				cv2.waitKey(1)
				continue

			projection = projection_matrix(camera_matrix, MH)
			
			if projection.any():
				

				currentFrame, points_memory = self.render(currentFrame, obj, projection, points_memory)#render test(currentFrame, imgpts)

				cv2.imshow('webcam', currentFrame)
				cv2.waitKey(1)
				
			else:
				cv2.waitKey(1) 
				continue


class Matcher:

	def __init__(self, selected_frame, alg, distCoeffs, camera_matrix, test_param):
		self.selected_frame = selected_frame
		self.alg = alg
		self.camera_matrix = camera_matrix
		self.distCoeffs = distCoeffs
		self.image = self.selected_frame.getImage_G()
		self.test_param = str(test_param)
		self.MH_memory = []
		

	def setFrame(self, frame):
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		self.frame = cv2.GaussianBlur(frame_gray, (3,3), 0)
		#self.frame = cv2.cvtColor(cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15), cv2.COLOR_BGR2GRAY)

	def getCorrespondence(self):
		kp1 = self.selected_frame.keypoints
		des1 = self.selected_frame.descriptors

		if self.alg == '1':
			orb = cv2.ORB_create()#(nfeatures = 1000, scaleFactor = 1.1, scoreType=cv2.ORB_FAST_SCORE)
			kp2, des2 = orb.detectAndCompute(self.frame, None) # kp2: keypoints of captured frame
			FLANN_INDEX_LSH = 6
			index_params= dict(algorithm = FLANN_INDEX_LSH,
				   table_number = 12,
				   key_size = 20,
				   multi_probe_level = 2)

		elif self.alg == '2':
			FLANN_INDEX_KDTREE = 1
			sift = cv2.xfeatures2d.SIFT_create()
			kp2, des2 = sift.detectAndCompute(self.frame, None)
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)


		search_params = dict(checks = 100)
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		matches = flann.knnMatch(des1,des2,k=3) 

		good = []
		for mn in matches:
			if len(mn) != 3: # not always 2 points (error) https://stackoverflow.com/questions/25018423/opencv-python-error-when-using-orb-images-feature-matching
				continue
			(m, n, a) = mn
			if ( (m.distance < 0.7*n.distance) and (m.distance < 0.7*a.distance) ):
				good.append(m)


		if len(good)>MIN_MATCH_COUNT:
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


			MH, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0) #0 regular all points | RANSAC | LMEDS
			np.matrix.round(MH, decimals=-1)
			if smoothing == True:
				if len(self.MH_memory) > 10:
					self.MH_memory.insert(0,MH)
					MH = sum(self.MH_memory[:-9]) / 9
					self.MH_memory.pop()

					#memor.append(projection)
				else:
					self.MH_memory.insert(0,MH)
					#memor.append(projection)


			matchesMask = mask.ravel().tolist()

			h,w = self.selected_frame.image.shape 
			pts = np.float32([ [0,0] , [0,h-1] , [w-1,h-1] , [w-1,0] ]).reshape(-1,1,2) # 4 vertices

			if self.test_param == '1':
				img2 = self.image
				img1 = self.frame
				img3 = cv2.drawMatches(img2, kp1, img1, kp2, good,None, flags=2)
				cv2.imshow('Match test',img3)
				cv2.waitKey(1)

			try:
				'''
				<class 'numpy.ndarray'>
				[[[391.25125 144.29417]]

				 [[394.3986  224.69884]]

				 [[592.92883 224.11461]]

				 [[594.6703  145.18074]]]

				'''
				

				corners = cv2.perspectiveTransform(pts,MH)

			except:
				print('No points (homography)')
				return

			print('Found: ',len(good))
			
			

			return (src_pts, dst_pts, corners, MH)

		else:
			print("Not enough matches are found:", len(good), "/", MIN_MATCH_COUNT)
			matchesMask = None
			return None



# projection_matrix = kamera_parameters * extrinistic parameters
#camera_matrix
#[fx 0  x0]
#[0  fy y0]
#[0  0  1]
# camera_mat^(-1) * homography = R1, R2, t
#rotation 3x3
#translation 3x1
def projection_matrix(camera_parameters, homography):
	
			homography = homography * (-1)# all values negative(odbicie lustrzane)
			mul_hom_cam = np.matmul(np.linalg.inv(camera_parameters), homography)# inverse camera_par ^(-1) * homography
			column1 = mul_hom_cam[:, 0]
			column2 = mul_hom_cam[:, 1]
			column3 = mul_hom_cam[:, 2]
			l = math.sqrt(np.linalg.norm(column1, 2) * np.linalg.norm(column2, 2))# normalisation
			rot_1 = column1 / l #rotation x
			rot_2 = column2 / l #rotation y
			translation = column3 / l

			c = rot_1 + rot_2
			p = np.cross(rot_1, rot_2) # wektor prostopadły do rot1 i 2
			d = np.cross(c, p)

			rot_1 = c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2) * 1 / math.sqrt(2)
			rot_2 = c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2) * 1 / math.sqrt(2)
			rot_3 = np.cross(rot_1, rot_2)
			projection = np.stack((rot_1, rot_2, rot_3, translation)).T
			return np.dot(camera_parameters, projection)



def choose_alg():
	print("""Choose algorithm:
1 ORB
2 SIFT """)
	alg = input()
	if not alg in ('1', '2'):
		sys.exit(0)
	return alg

def choose_test_parameters():
	print("""Show match test?:
1 Yes
Other No """)
	test = input()
	if not test in ('1'):
		return 0
	return test


if __name__ == '__main__':
	obj = load_obj.Object_load('./kubek.obj') 
	alg = choose_alg()	
	test_param = choose_test_parameters()
	prog = Program(alg, test_param)
	prog.main()

