import pickle
from lib.ituils.ituils import *
from lib.show.testshow import videodemo,photodemo

def main(detectflag):
    texturefilters = loadmat('lib/ituils/ICAtextureFilters_7x7_8bit.mat')['ICAtextureFilters'][:]
    with open('model/clf_C_0.1_inter_10000_.pickle', 'rb') as f:
        clf = pickle.load(f)
    if detectflag == 0:
        detector = dlib.get_frontal_face_detector()
    elif detectflag == 1:
        detector = dlib.cnn_face_detection_model_v1('lib/ituils/mmod_human_face_detector.dat')
    else:
        raise("error")
    path = '../data/video/false_Trim3.mp4'
    videodemo(path,clf,detector,texturefilters,detectflag)
    photodemo(path,clf,detector,texturefilters,detectflag)
if __name__ == "__main__":
    detectflag = 0 #人脸检测器标志
    main(detectflag)
##模型预测过程中显示窗口，适合测试时观察
