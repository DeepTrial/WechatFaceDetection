import cv2
import os
import copy
import numpy as np

crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
	# 旋转角度的周期是360°
    angle %= 360
	# 用OpenCV内置函数计算仿射矩阵
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
	# 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))
    # 如果需要裁剪去除黑边
    if crop:
	    # 对于裁剪角度的等效周期是180°
        angle_crop = angle % 180
		
		# 并且关于90°对称
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
			
		# 转化角度为弧度
        theta = angle_crop * np.pi / 180.0
		
		# 计算高宽比
        hw_ratio = float(h) / float(w)
		
		# 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
		
		# 计算分母项中和宽高比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
		
		# 计算分母项
        denominator = r * tan_theta + 1
		
		# 计算最终的边长系数
        crop_mult = numerator / denominator
		
		# 得到裁剪区域
        w_crop = int(round(crop_mult*w))
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)
    return img_rotated


def DataAugment(dir_path):  
    num=0
    for fr in dir_path:  
        suff = fr.split('.')[1]  
        filename = fr  
        img = cv2.imread(filename)  
        rotate45=rotate_image(img,45,True)
        rotate180=rotate_image(img,180,True)
        rotate225=rotate_image(img,225,True)
        rotate75=rotate_image(img,75,True)
        new_name45 ="%s/rotat45e_%09d.jpg"%('./cache',num)  
        new_name180 ="%s/rotate180_%09d.jpg"%('./cache',num)
        new_name225 ="%s/rotate225_%09d.jpg"%('./cache',num)
        new_name75 ="%s/rotate75_%09d.jpg"%('./cache',num) 
        num+=1  
        cv2.imwrite("%s/%05d.jpg"%('./cache',num),img)
        cv2.imwrite(new_name45,rotate45)  
        cv2.imwrite(new_name180,rotate180) 
        cv2.imwrite(new_name225,rotate225) 
        cv2.imwrite(new_name75,rotate75)                
        #cv2.imshow('image',iLR)  
        #cv2.waitKey(0)  

def getFileName():
    dir=os.path.join('./image','cache')
    imagePath=[]
    for root,dirs,files in os.walk(dir):
        for file in files:
            imagePath.append(os.path.join(root,file))
    return imagePath


imagePath=getFileName()

DataAugment(imagePath)