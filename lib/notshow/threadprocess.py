import cv2 as cv
from lib.extract_feature import extract_faeture
def zxc(clf,frame,texturefilters,detector,detectflag):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rects = detector(gray,0)
    if len(face_rects)==0:  
        return None
    else:   
        for _, d in enumerate(face_rects):
            if detectflag == 0:
                x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            else:
                x1, y1, x2, y2 = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
            try:
                if frame.shape[0] != 64 or frame.shape[1]!=64:
                    im =  cv.resize(frame[y1:y2,x1:x2,:],(64,64))
                else : im = frame

                fea = extract_faeture(im,texturefilters)
                result = clf.predict(fea[0:1,:])
            except cv.error:
                result = None
                continue
    # print(len(face_rects))
    return result
