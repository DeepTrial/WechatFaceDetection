import os


def fileList(dirName):
    '''
    遍历目标文件夹
    返回要删除目录里的子目录和文件
    '''
    dir=os.path.join(os.getcwd(),dirName)
    filePath=[]
    dirPath=[]
    for root,dirs,files in os.walk(dir):
        for file in files:
            filePath.append(os.path.join(root,file))
        for dir in dirs:
            dirPath.append(os.path.join(root,dir))
    return filePath,dirPath

def clean_dir(dirName):
    '''
    删除函数
    删除dirName里的子文件夹和文件
    '''
    print("开始清理...")
    filePath,dirPath=fileList(dirName)
    for file in filePath:
        os.remove(file)
    for dir in dirPath:
        print(dir,dirPath)
        os.rmdir(dir)
    print("%s 完成清理"%dirName)

def clean_cache():
    '''
    删除cache缓存
    '''
    clean_dir('./face_find_cv/collectImage/image/cache')

def clean_audio():
    '''
    删除audio缓存
    '''
    clean_dir('./audio')

def clean_trainset():
    '''
    删除训练集
    '''
    clean_dir('./face_find_dl/train_image')
