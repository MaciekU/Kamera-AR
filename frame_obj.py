import sys
import os
import cv2
import numpy as np
import math
	
class Marker_frame(object):

	def __init__(self, image, alg):

		frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		self.image = cv2.GaussianBlur(frame_gray, (3,3), 0)
		#self.image = cv2.cvtColor(cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15), cv2.COLOR_BGR2GRAY)

		if alg == '1':
			orb = cv2.ORB_create()#(nfeatures = 1000, scaleFactor = 1.1, scoreType=cv2.ORB_FAST_SCORE) # default Harris score https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html

			self.keypoints, self.descriptors = orb.detectAndCompute(self.image, None)
		elif alg == '2':
			sift = cv2.xfeatures2d.SIFT_create()
			self.keypoints, self.descriptors = sift.detectAndCompute(self.image, None)
			
		width, height = self.image.shape

		maxSize = max(width, height)
		self.w = width/maxSize
		self.h = height/maxSize

		# corner points 
		self.points2d = np.array([[0,0],[width,0],[width,height],[0,height]]) 

		self.points3d = np.array([[0,0,0],[self.w,0,0],[self.w,self.h,0],[0,self.h,0]]) 

	def get_points2d(self): # not used 3d object only
		return self.points2d

	def get_points3d(self):
		return self.points3d
	
	def getImage_G(self):
		return self.image
	
	def get_wh(self):
		return self.w, self.h
	

