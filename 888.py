import cv2
a=cv2.imread('0.jpg')
cv2.imwrite('aaa.jpg',a[50:,20:])