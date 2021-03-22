import cv2 
import numpy as np
import scipy.fftpack  
import matplotlib.pyplot as plt
import os


def SEGMENT(img):
    
    rows = img.shape[0]
    cols = img.shape[1]
    img = img[:, 25:cols-20]
    rows = img.shape[0]
    cols = img.shape[1]
    
    imgLog = np.log1p(np.array(img, dtype="float") / 255)
    M = 2*rows + 1
    N = 2*cols + 1
    sigma = 15
    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
    Hhigh = 1 - Hlow

    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

    gamma1 = 0.5
    gamma2 = 1.5
    Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]

    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")

    Ithresh = Ihmf2 < 65
    Ithresh = 255*Ithresh.astype("uint8")
    
   
    k1=0
    maxim=0
    Ithresh=cv2.resize(Ithresh, (150, 180), interpolation = cv2.INTER_AREA)
    for i in range(149):
        for j in range(179):
            if j==149:
                continue
            else:
                if Ithresh[j,i]==0:
                    continue
                else:
                    if  k1==0:
                        k1=j
                    else:
                        if Ithresh[j+1,i]==0:
                            k2=j
                            k=k2-k1
                            k1=0
                            if k > maxim:
                                maxim=k
        
    
    numbers=[]
    ret,thresh1 = cv2.threshold(Ithresh,0,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
    	x,y,w,h = cv2.boundingRect(cnt)
    	
    	cv2.rectangle(Ithresh,(x,y),(x+w,y+h),(0,255,0),3)
    i=25
    for cnt in contours:
    	x,y,w,h = cv2.boundingRect(cnt)

    	if w>2 and h>maxim-(maxim/5):
    		
    		numbers.append(thresh1[y-5:y+h+5,x-5:x+w+5])
    		i=i+1
    
    t1=0
    t2=0
    for line in numbers:
        x,y=line.shape
        t1+=x
        t2+=y
    t1=t1/len(line)
    t2=t2/len(line)
    newNumbers=[]
    for line in numbers:
        x,y=line.shape
        if abs(x-t1)>100 or abs(y-t2)>100:
            continue
        else:
            newNumbers.append(line)
        
    return newNumbers , Ithresh 

