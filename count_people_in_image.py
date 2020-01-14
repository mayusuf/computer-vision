#===========================================================================================
#
#title           :Count persons in an image. Also in series of images.
#description     :This script count persons in an image. Also in series of images.
#author		 	 :Abu Yusuf
#date            :20190114
#version         :1.0    
#usage		     :count_people_in_image.py
#notes           :Install minimum python 2 or 3, Python
#
#============================================================================================


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

imageFolder = 'person_images'

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


def main():
    import sys
    from glob import glob
    import itertools as it
    import os
       
    
    hog = cv.HOGDescriptor()
    hog.setSVMDetector( cv.HOGDescriptor_getDefaultPeopleDetector() )

    os.chdir(os.getcwd())
    default = imageFolder+os.sep
    filename = os.listdir(default); # list the directories 
    
      
    total = 0
    
    for fn in filename:
        print(fn, ' - ',)
        try:
            img = cv.imread(default+fn)
            if img is None:
                print('Failed to load image file:', fn)
                continue
        except:
            print('loading error')
            continue

        found, _w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        draw_detections(img, found)
        draw_detections(img, found_filtered, 3)
        
        print('%d (%d) found' % (len(found_filtered), len(found)))
        
        total += len(found_filtered) # couting person in a frame or an image
        
        cv.imshow('img', img)
        ch = cv.waitKey()
        if ch == 27:
            break

    print('In total {} persons are present here.'.format(total))
    
    print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
