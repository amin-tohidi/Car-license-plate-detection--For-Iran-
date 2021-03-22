import os
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from segmention import SEGMENT
import matplotlib.pyplot as plt
from keras.models import load_model
import keras
import shutil
import numpy as np


Tk().withdraw() 
filename2 = askopenfilename() 
img1=cv2.imread(filename2, 1)
# img1=cv2.resize(800, 900)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

filtered = cv2.GaussianBlur(gray, (5,5), 0)

edged = cv2.Canny(filtered, 10, 100) 
cv2.imshow('2211', edged)            


contours, hir = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

plate = False
for c in contours:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approx) == 4 and cv2.contourArea(c) > 1000:
        x, y, w, h = cv2.boundingRect(c)
        if 2.5 < w / h < 4.1:
            plate = True
            cv2.drawContours(img1, c, -1, (0, 255, 0), 3)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 3)
            break
            
if not plate:       
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) >= 4:
             x, y, w, h = cv2.boundingRect(c)
             if 2.5 < w / h < 4.5 and 10000 <= (w * h):
                 plate = True
                 cv2.drawContours(img1, c, -1, (0, 0, 255), 1)
                 cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0,255), 3)
                 break
cv2.imshow('TASVIR ASLI', img1)
    
if plate:
    cropped = img1[y-10:y+h+25, x-10:x+w+30]
    cv2.imshow('PELAK', cropped)
    cv2.imwrite('plate.jpg', cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()    



for root, dirs, files in os.walk('C:/Users/AMIN/Desktop/PLT/SEG'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))

img=cv2.imread("C:/Users/AMIN/Desktop/PLT/plate.jpg", 0)

plt.imshow(img)

num,plate=SEGMENT(img)

plt.figure(figsize = (5,10))
plt.imshow(plate)

k=20
j=0
for i in num:

    ax=plt.subplot(10, 20, k+1)
    ax.axis('off')
    plt.imshow(i)
    k+=2
    # plt.figure(figsize = (10,10))
    # plt.imshow(i)
    cv2.imwrite('SEG/'  + 'AA'+str(j)+'.jpg', i)
    j += 1
    
# --------------------------------------------------------------------------------
    
model = keras.models.load_model('C:/Users/AMIN/Desktop/PLT/20.h5')

directory = 'C:/Users/AMIN/Desktop/PLT/SEG'
number_of_files = len([item for item in os.listdir(directory) 
                           if os.path.isfile(os.path.join(directory, item))])
print("از عدد های موجود در پلاک ",number_of_files ,"عدد شناسایی شد.")

p=[]
for i in range(0,number_of_files):
        
        img = cv2.imread('C:/Users/AMIN/Desktop/PLT/SEG/AA'+ str(i) +'.jpg',0)
        plt.imshow(img, cmap="Greys")
        img = cv2.resize(img, (60, 120))
        img = np.reshape(img, [1, 60, 120, 1])
        index =np.argmax(model.predict(img))
        p.append(index)
        i=i+1
        
img2 = cv2.imread("C:/Users/AMIN/Desktop/PLT/plate.jpg")
plt.imshow(img2, cmap="Greys")
    
print("نتیجه تشخیص اعداد شناسایی شده در پلاک به صورت زیر است    :    ")
print(p)

if len(p)>8:
    print("\n\n متاسفانه بعضی ازاشکالی که شبیه کاراکتر بودند، به اشتباه کاراکتر تشخیص داده شده اند")
    

