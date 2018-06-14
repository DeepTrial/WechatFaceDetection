import cv2
import sys
import os
import numpy as np

#def imageList_online():
    #dir=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)),'faceDetect_api/face_find_cv/collectImage/image/face_image')
    #imagePath=[]
    #for root,dirs,files in os.walk(dir):
        #for file in files:
            #imagePath.append(os.path.join(root,file))
    #return imagePath

def imageList_postive():
    '''
    正样本图片列表
    '''
    dir='./face_find_dl/train_image/'
    #print(dir)
    imagePath=[]
    for root,dirs,files in os.walk(dir):
        for file in files:
            imagePath.append(os.path.join(root,file))
    return imagePath

def imageList_negtive():
    '''
    负样本图片列表
    '''
    dir='./face_find_dl/passive_image/'
    imagePath=[]
    for root,dirs,files in os.walk(dir):
        for file in files:
            imagePath.append(os.path.join(root,file))
    return imagePath
    

def dataLoad_offline():
    '''
    加载训练集
    '''
    imageList=imageList_postive()                     #正样本列表
    passiveList=imageList_negtive()                   #负样本列表
    x_train=[]
    x_label=[]
    num1=0
    num2=0
    
    for i in range(min(len(imageList),len(passiveList))):         #正负样本 1:1混合
        if i%2==0:
            image=cv2.imread(imageList[num1])
            image=cv2.resize(image,(48,48))
            image=np.reshape(image,(48,48,3))
            x_train.append(image)
            x_label.append(np.eye(2)[1])
            num1=num1+1
        else:
            image=cv2.imread(passiveList[num2])
            image=cv2.resize(image,(48,48))
            image=np.reshape(image,(48,48,3))
            x_train.append(image)
            x_label.append(np.eye(2)[0])
            num2=num2+1
    x_train=np.asarray(x_train)
    x_label=np.asarray(x_label)
    return x_train,x_label

#def dataLoad_online():
    #imageList=imageList_offline()
    #x_train=[]
    #x_label=[]
    #image=cv2.imread(imageList[i])
    #image=cv2.resize(image,(48,48))
    #image=np.reshape(image,(48,48,3))
    #x_train.append(image)
    #x_label.append(np.eye(2)[1])
    #x_train=np.asarray(x_train)
    #x_label=np.asarray(x_label)
    #return x_train,x_label


dataLoad_offline()