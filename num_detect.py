from PIL import Image
import cv2
import numpy as np
import requests

img = cv2.imread("images2/036.jpg")

# resizing
scale_percent = 25 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# making an array of image
image_arr = np.array(resized)

grey = cv2.cvtColor(image_arr,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(grey,(5,5),0)

dilated = cv2.dilate(blur,np.ones((3,3)))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
cv2.imshow("Closing Image",closing)
cv2.waitKey(0)

car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
print(car_cascade)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)
print(cars)

cnt = 0
for (x,y,w,h) in cars:
    cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
    cnt += 1
print(cnt, " cars found")

cv2.imshow("Closing Image",image_arr)
cv2.waitKey(0)