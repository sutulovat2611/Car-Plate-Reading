import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
# import easyocr



img = cv2.imread("images/005.jpg")

img = imutils.resize(img, width=1000)
img = imutils.resize(img, height=1000)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
# gray = cv2.bitwise_not(gray)

bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
# cv2.imshow("bfilter",bfilter)
# cv2.waitKey(0)

edged = cv2.Canny(bfilter, 240, 255)
cv2.imshow("edged",edged)
cv2.waitKey(0)

squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
light = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, squareKern)
# cv2.imshow("stru",light)
# cv2.waitKey(0)


thresh = cv2.erode(light, None, iterations=1)
# cv2.imshow("erode",thresh)
# cv2.waitKey(0)

thresh = cv2.dilate(thresh, None, iterations=1)
cv2.imshow("dilate",thresh)
cv2.waitKey(0)


cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
smtg = cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
cv2.imshow("cont",img)
cv2.waitKey(0)


location = None
crop = None
for c in cnts:
    epsilon = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    _, _, w, h = cv2.boundingRect(c)
    print(w, h)
    print(len(approx))
    # approx = cv2.approxPolyDP(c, 10, True)
    #  len(approx) >= 2 and len(approx) < 6 and 
    if w/h >= 1 and w/h <=6 :
        location = approx
        break
        
if location is not None:
    mask = np.zeros(gray.shape, np.uint8)
    new_img = cv2.drawContours(mask, [location], 0, 255, -1)
    new_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("bitwise",new_img)
    cv2.waitKey(0)

    
    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    crop = gray[x1-5:x2+5, y1-5:y2+5]
    print(crop)

    
else:
    print("location not found")


blur = cv2.GaussianBlur(crop, (5,5), 0)

unsharp_image = cv2.addWeighted(crop, 2, blur, -1, 0)


cnts, _ = cv2.findContours(unsharp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

smtg = cv2.drawContours(unsharp_image, cnts, -1, (0, 255, 0), 2)
# plt.imshow(cv2.cvtColor(smtg, cv2.COLOR_BGR2RGB))
cv2.imshow("contour2",unsharp_image)
cv2.waitKey(0)




# # Showing cropped thresholded version in a plot
# unsharp_image = unsharp_image[1:] 
# crop2 = cv2.threshold(unsharp_image, 180, 220, cv2.THRESH_BINARY)[1] 
# plt.imshow(cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB))
# plt.show()

#cases that work: 001, 003, 004, 007, 006, 023, 026, 030, 033, 034, 037, 038, 039

#cases that dont: 000, 002, 005, 008, 009, 010, 011,  012, 013, 015, 016, 017, 018