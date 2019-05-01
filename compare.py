import numpy as np
import cv2 as cv
f1=cv.imread('sift_keypoints.jpg')
f2=cv.imread('sift_keypoints1.jpg')
f1 = cv.cvtColor(f1,cv.COLOR_BGR2GRAY)
f2 = cv.cvtColor(f2,cv.COLOR_BGR2GRAY)
[f,des1] = vl_sift(f1)
[fn,des2] = vl_sift(f2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
