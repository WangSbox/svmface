import dlib # dlib for accurate face detection
import cv2 as cv # opencv
import imutils # helper functions from pyimagesearch.com
import os
# Face detector
detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1('lib/ituils/mmod_human_face_detector.dat')
path = ['E:/DESKTOP/real/','E:/DeskTop/false/']
name = ['1jpg','0jpg']
for num in range(len(path)):
    count = 0
    if num == 0:        per_fr = 2 #采集间隔
    elif num == 1:        per_fr = 2
    for video in os.listdir(path[num]):
        #print(os.path.join(path[num],video))
        vfr = cv.VideoCapture(os.path.join(path[num],video))
        while 1:
            ret, frame = vfr.read()
            count += 1
            if frame is None:                break
            else:
                if count%per_fr==0:
                    fm = frame.copy()
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    alpha  = 0.5
                    # detect faces in the gray scale frame
                    face_rects = detector(gray,0)
                    if len(face_rects)==0:
                        cv.imshow('face',frame)
                        cv.waitKey(1) & 0xFF
                    else:
                        for i, d in enumerate(face_rects):
                            #x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()#hog feature
                            x1, y1, x2, y2 = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom() # cnn feature
                            cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2) 
                            cv.imshow('face',frame)
                            cv.waitKey(1) & 0xFF
                            cv.imwrite(os.path.join(name[num],str(int(count/per_fr))+'.jpg'),cv.resize(fm[y1:y2,x1:x2,:],(64,64)))
        vfr.release()
        cv.destroyAllWindows()