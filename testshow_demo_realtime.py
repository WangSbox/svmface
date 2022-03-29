from lib.ituils.ituils import *
from lib.extract_feature import extract_faeture

detectflag = 0
if detectflag == 0:
    detector = dlib.get_frontal_face_detector()
elif detectflag == 1:
    detector = dlib.cnn_face_detection_model_v1('lib/ituils/mmod_human_face_detector.dat')
texturefilters = loadmat('lib/ituils/ICAtextureFilters_7x7_8bit.mat')['ICAtextureFilters'][:]    
with open('model/clf_C_0.1_inter_10000_.pickle', 'rb') as f:
    clf = pickle.load(f)
    #测试读取后的Model
vfr = cv.VideoCapture(0)
zd,jd = 0,0
ct = 0
ppp = 2
while 1:
    #start = time.time()
    ret, frame = vfr.read()
    ct += 1
    if ct%ppp != 0:
        continue
    if frame is None:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    alpha  = 0.5
    face_rects = detector(gray,0)
    
    for i, d in enumerate(face_rects):
        if detectflag == 0:
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right(), d.bottom(), d.width(), d.height()
        else:
            x1, y1, x2, y2 = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
        im =  cv.resize(frame[y1:y2,x1:x2,:],(64,64))
        fea = extract_faeture(im,texturefilters)
        result = clf.predict(fea[0:1,:])
        if result==0: 
                restext = 'False'
                jd += 1
        else: 
                restext = 'True'
                zd += 1
        cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2) #for drawing rectangle on deteced face, circle ,line anything can be used in place of this
        cv.putText(frame,restext,(x1,y1),cv.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
        cv.imshow('face',frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
vfr.release()
cv.destroyAllWindows()

"""
实时检测使用
"""