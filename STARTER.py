import itchat,cv2,sys,os,requests
from itchat.content import *
import numpy as np

sys.path.append("./face_find_dl/")
import face_find_cv.faceDetector as cvFace
import face_find_dl.work as dlFace
import face_find_dl.data_argument as moreData
import api_connect,clean_memory

cvDetect=cvFace.faceOrientate()

global imageName        #当前处理的图片地址
global label            #当前处理的图片的正确类标
global autoCount        #缓存清除计数器 %100=0 时清缓存
global reupload         #重新上传训练集标志    True->重新上传 False->不重新上传
global uploadCount      #新训练集计数器
global opencvMark       #使用opencv做预判断的标志  1->使用  0->不使用
opencvMark=1            #默认使用opencv
autoCount=0
reupload=False
uploadCount=0
wrongCollect=['错误','落伍','咯','服务','火舞','说我','若无','朴','购物']   #模糊匹配集

def judge_wrong(msg):
    '''
    在线学习实现函数
    '''
    global imageName
    global label

    print("wrong")
    itchat.send_msg(u"开始在线学习...请稍后...",msg['FromUserName'])
    image=cv2.imread(imageName)
    image=cv2.resize(image,(48,48))                       #调整大小到  48x48
    image,label=dlFace.online_learning(image,label)       #调用数据增强 增加多样性
    dlFace.train_online(image,label)                      #启动在线学习
    itchat.send_msg(u"在线学习完成！",msg['FromUserName'])




def clear_memory(mode='auto',trainSet=False):
    '''
    清除缓存函数
    参数列表:mode: Type str,trainset:Type bool
    mode:运行模式 auto自动运行 每100条数据清除一次缓存
    trainset：删除训练数据
    '''
    global autoCount
    if mode=='auto' and autoCount%100==0:
        clean_memory.clean_cache()
        clean_memory.clean_audio()
    elif trainSet==True:
        clean_memory.clean_trainset()
    else:
        pass

def opencv_detector(image):
    '''
    opencv检测函数
    调用opencv预训练的人脸检测函数(传统图像方法：haar+boost的方法)
    参数：
        image: 3维图像数据
    '''
    global label
    result=cvDetect.image_face_orientate(image,imageShow=False,imageTest=True,saveMark=True)
    if result==1:
        #print("cvpass!")
        label=np.eye(2)[0]
        return True
    return False


@itchat.msg_register(itchat.content.TEXT)
def turling_reply(msg):
    '''
    文字信息处理函数
    一般对话调用图灵机器人api(自建知识库)自动回复
    特殊关键词：训练、错误、关闭等优先处理
    重新训练：重新训练深度学习模型
    错误：为防止语音识别错误，设置的后门，可以启动在线学习
    关闭opencv：关闭opencv人脸检测
    开启opencv: 开启opencv人脸检测
    删除训练集: 删除./face_find_cv/collectImage/image/train_image训练集
    完成上传：  完成重新上传训练集 开始训练
    '''
    global opencvMark
    global reupload
    global uploadCount

    if u'重新训练' in msg['Text']:
        itchat.send_msg(u"正在训练分类器...请稍后...",msg['FromUserName'])
        dlFace.train(50)
        itchat.send_msg(u"训练完成！",msg['FromUserName'])
    elif u'错误' in msg['Text']:
        judge_wrong(msg)
    elif u'关闭opencv' in msg['Text']:
        opencvMark=0
        itchat.send_msg(u"OpenCV已关闭!",msg['FromUserName'])
    elif u'开启opencv' in msg['Text']:
        opencvMark=1
        itchat.send_msg(u"OpenCV已开启!",msg['FromUserName'])
    elif u'删除训练集' in msg['Text'] or u'重新上传训练集' in msg['Text']:
        clear_memory(mode='manual',trainSet=True)
        reupload=True
        opencvMark=0
        itchat.send_msg(u"完成删除！OpenCV已关闭!\n开始上传训练集",msg['FromUserName'])
    elif u'完成上传' in msg['Text']:
        reupload=False
        uploadCount=0
        itchat.send_msg(u"正在训练分类器...请稍后...",msg['FromUserName'])
        dlFace.train(50)
        itchat.send_msg(u"训练完成！",msg['FromUserName'])
    else:
        defaultReply=u'我不知道该说些什么'
        reply = api_connect.get_response(msg['Text'])      #当图灵机器人无法做出回复时，默认回复defaultReply
        return reply or defaultReply


@itchat.msg_register([PICTURE])
def download_image(msg):
    '''
    图像检测函数
    收到的图像将缓存在/face_find_cv/collectImage/image/cache 文件夹中
    '''
    global imageName
    global label
    global opencvMark
    global autoCount
    global reupload
    global uploadCount

    if msg['Type']=='Picture':     #确认收到的类型是图片
        if reupload==False:
            clear_memory()
            imageName=os.path.join('./face_find_cv/collectImage/image/cache',msg.fileName)
            autoCount=autoCount+1
            msg.download(imageName)
            print(imageName)
            image=cv2.imread(imageName)
            itchat.send_msg(u"正在检测...请稍后...",msg['FromUserName'])
            result=opencv_detector(image)
            if opencvMark==1 and result==1:                                                #如果opencv选项关闭 则忽略此结果
                print("cvpass!")
                label=np.eye(2)[0]
                itchat.send_msg(u'OpenCV检测到人脸!',msg['FromUserName'])
            else:
                #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray=cv2.resize(image,(48,48))
                if dlFace.predict(gray)==1:
                    print("dlpass!")
                    label=np.eye(2)[0]
                    itchat.send_msg(u'深度学习模型检测到目标!',msg['FromUserName'])
                else:
                    print("no face!")
                    label=np.eye(2)[1]
                    itchat.send_msg(u'对不起!没有发现目标',msg['FromUserName'])
        else:
            imageName=os.path.join('./face_find_dl/train_image',msg.fileName)
            msg.download(imageName)
            img=cv2.imread(imageName)
            uploadCount=uploadCount+1
            itchat.send_msg(u'已上传%d张图片'%uploadCount,msg['FromUserName'])
            #图片数据增强
            moreData.rotate(img,45,online=False,dirName='./face_find_dl/train_image')
            moreData.rotate(img,180,online=False,dirName='./face_find_dl/train_image')
            moreData.rotate(img,225,online=False,dirName='./face_find_dl/train_image')
            moreData.rotate(img,315,online=False,dirName='./face_find_dl/train_image')


@itchat.msg_register([RECORDING])
def download_record(msg):
    '''
    语音处理函数
    缓存语音至./audio 格式mp3
    '''
    global imageName
    global label

    msg.download(os.path.join('./audio',msg.fileName))
    api_connect.mp3_2_wav(os.path.join('./audio',msg.fileName),'./audio/audio_convert.wav')    #将mp3格式换为wav格式
    word=api_connect.record_api()                                      #调用语音识别api
    if word in wrongCollect:
        judge_wrong(msg)
    else:
        if word==u'正确':
            print("right")
            itchat.send_msg(u"真幸运！",msg['FromUserName'])
        else:
            print("not know")
            itchat.send_msg(u"对不起我没听明白",msg['FromUserName'])


itchat.auto_login()
itchat.run()
