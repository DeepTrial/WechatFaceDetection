import cv2
import time
import numpy as np
import os

crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]


def rotate_image(img, angle, crop):
    '''
    数据增强(旋转操作)
    对需要在线学习的图像
    '''
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


def rotate(img,angle,online=True,dirName=None):
    '''
    数据增强入口函数
    img：输入图像
    angle：旋转的角度
    online：在线学习标志
    dirName：保存地址
    '''
    filename=int(time.time())

    image=rotate_image(img,angle,True)  
    if online==True:
        image=cv2.resize(image,(48,48))
        image=np.reshape(image,(48,48,3))
        return image
    else:
        cv2.imwrite(os.path.join(dirName,"new_%d_%d.jpg"%(angle,filename)),image)
        return True