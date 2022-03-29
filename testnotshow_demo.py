from lib.ituils.ituils import *
from lib.notshow.photopredict import predictphoto
from lib.notshow.videopredict import predictvideo
# 训练并保存模型

def main():
    detectflag = 0
    photopath = '../data/photo1'
    # phpath = ['../photo','../photo2']
    phpath = ['../data/photo2','../data/photo3','../data/photo4']
    videopath = '../data/video'

    texturefilters = loadmat('lib/ituils/ICAtextureFilters_7x7_8bit.mat')['ICAtextureFilters'][:]
    
    if detectflag == 0:
        detector = dlib.get_frontal_face_detector()
    elif detectflag == 1:
        detector = dlib.cnn_face_detection_model_v1('lib/ituils/mmod_human_face_detector.dat')
    with open('model/clf_C_0.1_inter_10000_.pickle', 'rb') as f:
        clf = pickle.load(f)
    

    predictphoto(clf,photopath,texturefilters,per_frame=2,num_worker=8,detectflag=detectflag)
    # for pppath in phpath:
        # funcp2(clf,pppath,detector,texturefilters,per_frame=1,num_worker=8)
    predictphoto(clf,phpath[0],detector,texturefilters,per_frame=1,num_worker=8,detectflag=detectflag)
    predictphoto(clf,phpath[1],texturefilters,per_frame=1,num_worker=8,detectflag=detectflag)
    predictphoto(clf,phpath[2],texturefilters,per_frame=1,num_worker=8,detectflag=detectflag)
    predictvideo(clf,videopath,texturefilters,detector,per_frame=4,num_worker=6,detectflag=detectflag)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))    

if __name__ == "__main__":
    main()
    ##模型直接出结果，并不显示窗口
