import numpy as np
import cv2 as cv
img = cv.imread('index.jpeg')
#gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#print(gray)
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(img,None)
img1=cv.drawKeypoints(img,kp,img)
cv.imwrite('sift_keypoints.jpg',img1)
