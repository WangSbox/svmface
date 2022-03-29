import os
import time
import multiprocessing
import numpy as np
import cv2 as cv
from .threadprocess import zxc

def predictphoto(clf,photopath,detector,texturefilters,per_frame=3,num_worker=8,detectflag=0):
    r'''
    clf:预测器
    photopath:照片存放文件的上层路径 该路径下包含真或假的对应的文件夹
    texturefilters:
    per_frame:照片特征提取间隔数
    num_worker:进程池数量
    detectflag:检测器标志位
    '''
    photo = os.listdir(photopath)
    for i in range(len(photo)):
        begin = time.time()
        pool = multiprocessing.Pool(processes=num_worker)
        result = []
        photo_list = os.listdir(os.path.join(photopath,photo[i]))
        for j in range(0,len(photo_list),per_frame):
            frame = cv.imread(os.path.join(os.path.join(photopath,photo[i]),photo_list[j]),1)
            if frame is  not None:
                # im = cv.resize(frame,(64,64))
                result.append(pool.apply_async(zxc, (clf,frame,texturefilters,detector,detectflag, )))
        pool.close()
        pool.join()
        res = []
        for re in result:
            yy = re.get()
            if yy == 0 or yy ==1:
                res.append(yy)
        res = np.array(res)
        end = time.time()
        if photo[i] == '0':
            print('path:{:<2},Time cost:{:.2f}, {:<4d}, {:<4d}, {:.4f}, {}'.format(photo[i],(end-begin)/60,np.sum(res==0),res.shape[0],  np.sum(res==0)/(res.shape[0]+1e-8),"Sub-process(es) done."))
        elif photo[i] == '1':
            print('path:{:<2},Time cost:{:.2f}, {:<4d}, {:<4d}, {:.4f}, {}'.format(photo[i],(end-begin)/60,np.sum(res==1),res.shape[0],  np.sum(res==1)/(res.shape[0]+1e-8),"Sub-process(es) done."))
        else:
            print('path:{:<2},Time cost:{:.2f}, {:<4d}, {:<4d}, {:.4f}, {}'.format(photo[i][:2],(end-begin)/60,np.sum(res==0),res.shape[0],  np.sum(res==0)/(res.shape[0]+1e-8),"Sub-process(es) done."))

    pool.close()
    pool.join()
