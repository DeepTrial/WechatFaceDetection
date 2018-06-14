import numpy
import cv2
import os
import time


class faceOrientate:
    face_cascade = cv2.CascadeClassifier('./face_find_cv/haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('./face_find_cv/haarcascade_eye_tree_eyeglasses.xml')
#face_detector
    def image_face_orientate(self,imageRead,imageShow=False,imageTest=True,saveMark=True):
        faceMark=0
        newtimestamp=int(time.time()*1000)
        lasttimestamp=0
        img = imageRead
        cv2.waitKey()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray=cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.2,5)
        for (x,y,w,h) in faces:
            faceMark=1
            face_found=img[int(y):int(y+h), int(x):int(x+w)]
            newtamp=int(time.time()*1000)
            if newtimestamp-lasttimestamp>=9000 and w>=img.shape[0]/8 and h>=img.shape[1]/8 and saveMark:
                saveType=cv2.resize(face_found,(48,48))
                cv2.imwrite(os.path.join("./face_find_cv/collectImage","image","face_cache","face_%d.jpg"%newtimestamp),saveType)
                lasttimestamp=newtimestamp
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        #eyes_detector
            roi_gray = gray[int(y):int(y+h/2), int(x):int(x+w)]
            roi_color = img[int(y):int(y+h/2), int(x):int(x+w)]
            eyes = self.eye_cascade.detectMultiScale(roi_gray,1.1,5)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        if imageShow==True:
            cv2.imshow('img',img)
            if imageTest==True:
                cv2.waitKey()
        if faceMark==1:
            return 1
        else:
            return 0

    def realTime_face_orientate(self):
        cam = cv2.VideoCapture(0)
        while True:
            ret,frame=cam.read()
            self.image_face_orientate(frame,imageShow=True,imageTest=False)
            cv2.waitKey(1)


#if __name__=="__main__":
    #dector=faceOrientate()
    #dector.realTime_face_orientate()
    #img=cv2.imread("1.jpg")
    #image_face_orientate(img)


    


