# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:30:02 2020

@author: lilliloo
"""

# -------------------------------- #
import cv2
import numpy as np
# -------------------------------- #

# Read Image(RGB)
img = cv2.imread("ghost.png")
# Read Image(gray)
#img_g = cv2.imread("ghost.png", 0)
# Read Image(RGBA)
#img_rgba = cv2.imread("ghost.png", -1)


# Print Image pixel
#print("RGB : ", img)

# Write Image
#cv2.imwrite("output1.jpg", img)

# Display Image
#cv2.imshow("Input", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# Get Image's informations
height, width, ch = img.shape

# Transform HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite("output_hsv.jpg", hsv)

# Musking
# 赤色の検出
def detect_red_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([0,64,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,64,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色領域のマスク（255：赤色、0：赤色以外）    
    mask = mask1 + mask2

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

# 緑色の検出
def detect_green_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 緑色のHSVの値域1
    hsv_min = np.array([30, 64, 0])
    hsv_max = np.array([90,255,255])

    # 緑色領域のマスク（255：赤色、0：赤色以外）    
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    
    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

# 青色の検出
def detect_blue_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 青色のHSVの値域1
    hsv_min = np.array([90, 64, 0])
    hsv_max = np.array([150,255,255])

    # 青色領域のマスク（255：赤色、0：赤色以外）    
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

# 色検出（赤、緑、青）
red_mask, red_masked_img = detect_red_color(img)
green_mask, green_masked_img = detect_green_color(img)
blue_mask, blue_masked_img = detect_blue_color(img)
# 結果を出力
cv2.imwrite("red_mask.png", red_mask)
cv2.imwrite("red_masked_img.png", red_masked_img)
cv2.imwrite("green_mask.png", green_mask)
cv2.imwrite("green_masked_img.png", green_masked_img)
cv2.imwrite("blue_mask.png", blue_mask)
cv2.imwrite("blue_masked_img.png", blue_masked_img)

# mozaik
# 窓画像の左上座標
x, y = 50, 100
# 窓画像の幅・高さ
w, h = 40, 40
# 窓画像を黒塗り(画素値を0に)
img[y:y+h, x:x+w] = 0
cv2.imwrite("output_p.jpg",img)






