from __future__ import print_function
from scipy.io import loadmat
from lib.extractfeaturefromfile.extractphoto import photoextract
from lib.extractfeaturefromfile.extractvideo import videoextract
def main():
    texturefilters = loadmat('lib/ituils/ICAtextureFilters_7x7_8bit.mat')['ICAtextureFilters'][:]
    path = r'E:\MATLABcx\faceup/data/'
    photoextract(path,texturefilters,per_frame=4,num_worker=2,save=True,save_perframe=5000) #视频人脸特征
    videoextract(path,texturefilters,per_frame=4,num_worker=2,save=True,save_perframe=5000) #照片人脸特征提取
    
if __name__ == "__main__":
    main()
    ##提取检测到的人脸特征并保存