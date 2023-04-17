# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:19:32 2023
将将贝叶棕和糖棕框取变成40*40的小图片
@author: admin
"""

import numpy as np
import argparse
import cv2
import glob
import os
def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized

image = cv2.imread(r'C:\Users\admin\Desktop\palm leaf of beiye\obvious\BG193_001_1.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()


image = resize(orig, height = 500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# 遍历轮廓
for c in cnts:
	# 计算轮廓近似
	peri = cv2.arcLength(c, True)
	# C表示输入的点集
	# epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
	# True表示封闭的
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# 4个点的时候就拿出来
	if len(approx) == 4:
		screenCnt = approx
		break
print("STEP 2: 获取轮廓")
cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#最大内接矩形
def order_points(pts):
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype = "float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序坐标(依次为左上右上右下左下)
    return rect
 
 
img = cv2.imread(r'F:\googledownload\WPM\all\DSC01237.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.blur(gray, (3, 3))
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
_, cnts, _ = cv2.findContours( thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("Edged", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
 
rect = order_points(c.reshape(c.shape[0], 2))
print(rect)
xs = [i[0] for i in rect]
ys = [i[1] for i in rect]
xs.sort()
ys.sort()
#内接矩形的坐标为
print(xs[1],xs[2],ys[1],ys[2])

cv2.drawContours(img, [rect.astype(int)], -1, (0, 255, 0), 2)
cv2.imshow("Outline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 透视变换
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
warped = four_point_transform(sobelxy, rect.astype(int))
cv2.imshow("Outline", sobelxy)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Outline", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
#设置一个选取的正方形纹理大小，贝叶棕的框取
vein_size = 80#选取框边长
path  = r'C:\Users\admin\Desktop\palm leaf of beiye\obvious'
savepath = r'F:\palmleaf_train\qige_80_color\byz'
filenames = os.listdir(path)
for fn in filenames:
    #print(os.path.join(path,fn))
    newpath = os.path.join(path,fn)
    img = cv2.imread(newpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (3, 3))
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    _, cnts, _ = cv2.findContours( thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = order_points(c.reshape(c.shape[0], 2))
    # sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    # sobelx = cv2.convertScaleAbs(sobelx)
    # sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    # sobely = cv2.convertScaleAbs(sobely)
    # sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    warped = four_point_transform(img, rect.astype(int))#切割原始图像还是边缘检测的图像在这里修改，img为原始图像，sobelxy为边缘检测的图像

    for i in range(int(warped.shape[0]/vein_size)):
        for j in range(int(warped.shape[1]/vein_size)):
            xqk = warped[i*vein_size:i*vein_size+vein_size,j*vein_size:j*vein_size+vein_size] #h=40,w=40,xqk选取框
            name = str(i)+'_'+str(j)+'_'+fn
        
            cv2.imwrite(os.path.join(savepath,name), xqk)
            
#糖棕的框取            
vein_size = 80
path  = r'F:\googledownload\WPM\all'
savepath = r'F:\palmleaf_train\qige_80_color\tz'
filenames = os.listdir(path)
for fn in filenames:
    #print(os.path.join(path,fn))
    newpath = os.path.join(path,fn)
    img = cv2.imread(newpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (3, 3))
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    _, cnts, _ = cv2.findContours( thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = order_points(c.reshape(c.shape[0], 2))
    # sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    # sobelx = cv2.convertScaleAbs(sobelx)
    # sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    # sobely = cv2.convertScaleAbs(sobely)
    # sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    warped = four_point_transform(img, rect.astype(int))

    for i in range(int(warped.shape[0]/vein_size)):
        for j in range(int(warped.shape[1]/vein_size)):
            xqk = warped[i*vein_size:i*vein_size+vein_size,j*vein_size:j*vein_size+vein_size] #h=40,w=40,xqk选取框
            name = str(i)+'_'+str(j)+'_'+fn
        
            cv2.imwrite(os.path.join(savepath,name), xqk)

    

path  = r'F:\palmleaf_train\palm leaf qiege2'
savepath = r'F:\palmleaf_train\tz8000'
filenames = os.listdir(path)
for fn,i in zip(filenames,range(8000)):
    #print(os.path.join(path,fn))
    newpath = os.path.join(path,fn)
    img = cv2.imread(newpath)
    cv2.imwrite(os.path.join(savepath,fn), img)

    

path1 =   r'C:\Users\admin\Desktop\palm leaf of beiye\obvious\BG041_001_1.jpg'
vein_size = 80
img = cv2.imread(path1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.blur(gray, (3, 3))
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
_, cnts, _ = cv2.findContours( thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
rect = order_points(c.reshape(c.shape[0], 2))
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
warped = four_point_transform(img, rect.astype(int))

    
cv2.imshow("Outline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

    
    