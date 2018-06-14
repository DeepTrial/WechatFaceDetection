import model,imageNorm,data_argument
import cv2
import numpy as np


global detor                                                  
detor=model.faceDetect()                                         #公共的model 节省内存
crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

def train(batch_size,redo=True):
    '''
    训练模型
    '''
    global detor
    x_train,x_label=imageNorm.dataLoad_offline()
    #detor=model.faceDetect()
    detor.train(x_train,x_label,50,batch_size,redo)

def predict(image):
    '''
    对图像做预测
    image:需要预测的图像 
    '''
    global detor
    #detor=model.faceDetect()
    test=np.reshape(image,(1,48,48,3))
    return detor.predict(test)

def train_online(x_train,label,redo=False):
    '''
    在线学习的训练函数
    '''
    global detor
    #detor=model.faceDetect()
    detor.train(x_train,label,4,5,redo)




def online_learning(image,label):
    '''
    在线学习入口
    image：在线学习的图像
    label：图像对应的真实类标
    '''
    x_label=[]
    x_train=[]
    # 1张图片增强为5张
    image45=data_argument.rotate(image,45)                 #旋转45度
   
    image180=data_argument.rotate(image,180)               #旋转180度

    image225=data_argument.rotate(image,225)               #旋转225度

    image315=data_argument.rotate(image,315)               #旋转315度

    x_train.append(image)
    x_train.append(image45)
    x_train.append(image180)
    x_train.append(image225)
    x_train.append(image315)

    for i in range(5):
        x_label.append(label)
    x_train=np.asarray(x_train)
    x_label=np.asarray(x_label)
    return x_train,x_label


#train(50)

#image=cv2.imread('1.png')
#image=cv2.resize(image,(48,48))

#image=np.reshape(image,(1,48,48,3))
#print(detor.predict_prob(image))
#label=np.eye(2)[0] 
#label=np.reshape(label,(1,2))
#train_online(image,label)
