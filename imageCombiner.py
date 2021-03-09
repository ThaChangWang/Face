import os 
import cv2 
import numpy as np
from PIL import Image 

def getint(name):
    num, ext = name.split('.')
    return int(num)

folder = 'final'
video_name = 'final.avi'

images = []
for filename in sorted(os.listdir(folder), key=getint):
    img = cv2.imread("final/" + filename)
    print(filename)
    if img is not None:
        images.append(img)



# setting the frame width, height width 
# the width, height of first image 
height, width, layers = images[0].shape 

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height)) 

# Appending the images to the video one by one 
for image in images:
	video.write(image) 

cv2.destroyAllWindows() 
video.release() 
