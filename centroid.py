#===========================================================================================
#
#title           : measure center point.
#description     : This script measure center point of object in an image.
#author		 	 :Abu Yusuf
#date            :20190114
#version         :1.0    
#usage		     :centroid.py
#notes           :Install minimum python 3
#
#============================================================================================



import numpy as np
import cv2

img = cv2.imread('images/the-shape.png')

# convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image,127,255,0)
 
# find contours in the binary image
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

########### BEGIN : For multiple object in an image #########################

for c in contours:
   # calculate moments for each contour
   M = cv2.moments(c)
 
   # calculate x,y coordinate of center
   if(M["m00"] == 0):
     a=1
   else:
     a = M["m00"]
     
   cX = int(M["m10"] /a)
   cY = int(M["m01"] /a)
   cv2.circle(img, (cX, cY), 5, (77,14,133), -1)
   #cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
   # display the image
   cv2.imshow("Image", img)
   cv2.waitKey(0)

########### END : For multiple object in an image #########################


########### BEGIN : For single object in an  image #########################
M = cv2.moments(thresh)
if(M["m00"] == 0):
    a=1
else:
    a = M["m00"]
     
cX = int(M["m10"] /a)
cY = int(M["m01"] /a)
   
#print(cX,cY)


# put text and highlight the center
cv2.circle(img, (cX, cY), 4, (77,14,133), -1)
 
# display the image
cv2.imshow("Image", img)
cv2.waitKey(0)

########### END : For single object in an  image #########################
