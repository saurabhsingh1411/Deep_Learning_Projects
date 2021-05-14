#Importing Lb
import numpy as np
import cv2
import matplotlib.pyplot as plt

#%matplotlib inline

#loading the image to dataset
test_image=cv2.imread('photo.JPG')

print(test_image.shape)

#cv2.imshow('test_image final',test_image)

#cv2.waitKey(delay=0)


#Converting to grey scale

test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

plt.imshow(test_image,cmap='gray')

def convertToRGB(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

haar_cascade_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

print(type(haar_cascade_face))

faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5)

print(len(faces_rects))

for x,y,w,h in faces_rects:
    cv2.rectangle(test_image,(x,y),(x+w,y+h),(0,255,0),2)

#convert image to RGB and show image
cv2.imshow('test_image final',convertToRGB(test_image))

cv2.waitKey(delay=0)