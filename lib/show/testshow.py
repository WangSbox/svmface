import cv2 as cv
from lib.ituils.ituils import *
from lib.extract_feature import extract_faeture
def videodemo(path,clf,detector,texturefilters,detectflag):
    r'''
    path:包含文件名的文件路径
    clf:预测器
    detector:人脸检测器
    texturefilters:
    '''
    vfr = cv.VideoCapture(path)
    #vfr = cv.VideoCapture('E:/DESKTOP/test/WeChat_20210928143012.mp4')
    zd, jd = 0, 0
    ct = 0
    ppp = 1
    while 1:
        #start = time.time()
        _, frame = vfr.read()
        ct += 1
        if ct%ppp != 0: continue
        if frame is None:   break
        begin = time.time()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        alpha  = 0.5
        face_rects = detector(gray,0)
        if len(face_rects)==0:
            cv.imshow('face',frame)
        else:
            for i, d in enumerate(face_rects):
                if detectflag == 0:
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right(), d.bottom(), d.width(), d.height()
                else:
                    x1, y1, x2, y2 = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
                im =  cv.resize(frame[y1:y2,x1:x2,:],(64,64))
                fea = extract_faeture(im,texturefilters)
                #result = clf.predict(fea[0:1,:])[0]
                result = clf.predict(fea[0:1,:])
                #print(result)
                if result==0: 
                    restext = 'False'
                    jd += 1
                else: 
                    restext = 'True'
                    zd += 1
                cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2) #
                cv.putText(frame,restext,(x1,y1),cv.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
                cv.imshow('face',frame)
        if cv.waitKey(1) & 0xFF==ord('q'):
            break
        end = time.time()
        print((end-begin))
    vfr.release()
    cv.destroyAllWindows()
    print('真:{:5f},假:{:5f}'.format(zd/(zd+jd),jd/(zd+jd)))
def photodemo(path,clf,detector,texturefilters,detectflag):
    r'''
    path:包含照片的文件夹路径
    clf:预测器
    detector:人脸检测器
    texturefilters:
    '''
    photo = os.listdir(path)
    zd, jd = 0, 0
    ct = 0
    ppp = 1
    for file in photo:
        #start = time.time()
        frame = cv.imread(os.path.join(path,file),1)
        ct += 1
        if ct%ppp != 0: continue
        if frame is None:   break
        begin = time.time()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        alpha  = 0.5
        face_rects = detector(gray,0)
        if len(face_rects)==0:
            cv.imshow('face',frame)
        else:
            for i, d in enumerate(face_rects):
                if detectflag == 0:
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right(), d.bottom(), d.width(), d.height()
                else:
                    x1, y1, x2, y2 = d.rect.left(), d.rect.top(), d.rect.right() + 1, d.rect.bottom() + 1
                im =  cv.resize(frame[y1:y2,x1:x2,:],(64,64))
                fea = extract_faeture(im,texturefilters)
                result = clf.predict(fea[0:1,:])
                if result==0: 
                    restext = 'False'
                    jd += 1
                else: 
                    restext = 'True'
                    zd += 1
                cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2) #
                cv.putText(frame,restext,(x1,y1),cv.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
                cv.imshow('face',frame)
        if cv.waitKey(1) & 0xFF==ord('q'):
            break
        end = time.time()
        print((end-begin))
    cv.destroyAllWindows()
    print('真:{:5f},假:{:5f}'.format(zd/(zd+jd),jd/(zd+jd)))