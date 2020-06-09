# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:14:49 2020

@author: lilliloo
"""

# -------------------------------- #
import cv2
import numpy as np
from matplotlib import pyplot as plt
# -------------------------------- #

# Read Image(RGB)
img = cv2.imread("ghost.png")

b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]

# 1(NumPyでヒストグラムの算出)
hist_r, bins = np.histogram(r.ravel(),256,[0,256])
hist_g, bins = np.histogram(g.ravel(),256,[0,256])
hist_b, bins = np.histogram(b.ravel(),256,[0,256])

#2(OpenCVでヒストグラムの算出)
#hist_r = cv2.calcHist([r],[0],None,[256],[0,256])
#hist_g = cv2.calcHist([g],[0],None,[256],[0,256])
#hist_b = cv2.calcHist([b],[0],None,[256],[0,256])

# グラフの作成
plt.xlim(0, 255)
plt.plot(hist_r, "-r", label="Red")
plt.plot(hist_g, "-g", label="Green")
plt.plot(hist_b, "-b", label="Blue")
plt.xlabel("Pixel value", fontsize=20)
plt.ylabel("Number of pixels", fontsize=20)
plt.legend()
plt.grid()
plt.show()