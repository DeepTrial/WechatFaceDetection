from keras.models import Sequential,model_from_json
from keras.layers.core import Flatten, Dense, Dropout,Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,Adam
from keras.callbacks import EarlyStopping
import os, argparse 
import keras.backend as K
import h5py 
import numpy
import sys
sys.path.append("./face_find_dl/")


#深度学习模型类
class faceDetect:
    learning_rate = 0.0001
    n_classes = 2
    n_fc1 = 4096
    n_fc2 = 2048

    def VGG_19(self,weights_path=None,redo=False):
        '''
        VGG19模型定义
        '''
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(48,48,3)))
        model.add(Convolution2D(64, 3, 3,activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))


        model.add(Flatten())
        model.add(Dense(self.n_fc1, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_fc2, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))
        #载入训练好的参数
        if weights_path and redo==False:
            model.load_weights(weights_path)

        return model

    def train(self,X_train,y_train,training_iters,your_batch_size,redo):
        '''
        训练模型
        '''
        model=self.VGG_19(os.path.join('./face_find_dl/data','classifier_weights.h5'),redo)
        #sgd = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        #优化函数 adam
        adam=Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #使用自停止方法 当loss不再减小时立刻停止训练
        earlystop=EarlyStopping(monitor='loss', patience=0, verbose=2, mode='min')
        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
        #开始训练
        model.fit(X_train,y_train,nb_epoch=training_iters,batch_size=your_batch_size,verbose=2,callbacks=[earlystop])
        #将模型结构保存在json文档中
        json_string = model.to_json()  
        open(os.path.join('./face_find_dl/data','classifier.json'),'w').write(json_string)  
        #保存学习的权重
        model.save_weights(os.path.join('./face_find_dl/data','classifier_weights.h5'))
        #释放内存
        K.clear_session()

    def predict(self,X_test):
        '''
        预测函数 预测X_test所属的类别
        '''
        #载入模型结构和参数
        model = model_from_json(open(os.path.join('./face_find_dl/data','classifier.json')).read())  
        model.load_weights(os.path.join('./face_find_dl/data','classifier_weights.h5'))
        result=model.predict_classes(X_test)
        #释放内存
        K.clear_session()
        return result
    
