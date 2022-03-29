from __future__ import print_function
import os
import time
import multiprocessing
import numpy as np
import cv2 as cv
from .threadprocess import zxc

def predictvideo(clf,videopath,texturefilters,detector,per_frame=3,num_worker=8,detectflag=0):
    r'''
    clf:预测器
    videopath:视频存放文件的上层路径 该路径下包含真或假的对应的文件夹
    texturefilters:
    per_frame:照片特征提取间隔数
    num_worker:进程池数量
    detectflag:检测器标志位
    '''
    video_list = os.listdir(videopath)
    for i in range(len(video_list)):
        begin = time.time()
        pool = multiprocessing.Pool(processes=num_worker)
        result = []
        vrf = cv.VideoCapture(os.path.join(videopath,video_list[i]))
        lf = int(vrf.get(7)) # vef.get(cv.CAP_PROP_FRAME_COUNT)
        for j in range(0,lf,per_frame):
            vrf.set(cv.CAP_PROP_POS_FRAMES,j)
            _,frame = vrf.read()
            if frame is  not None:
                result.append(pool.apply_async(zxc, (clf,frame,texturefilters,detector,detectflag, )))
        pool.close()
        pool.join()
        vrf.release()
        res = []
        for re in result:
            if re.get() != None:
                res.append(re.get())
        res = np.array(res)
        end = time.time()
        if video_list[i] == 'true.mp4':
            print('path:{:<2},Time cost:{:.2f}, {:<4d}, {:<4d}, {:.4f}, {}'.format(video_list[i][0:2],(end-begin)/60,np.sum(res==1),res.shape[0],np.sum(res==1)/res.shape[0],  "Sub-process(es) done."))
        else:
            print('path:{:<2},Time cost:{:.2f}, {:<4d}, {:<4d}, {:.4f}, {}'.format(video_list[i][-6:-4],(end-begin)/60,np.sum(res==0),res.shape[0],np.sum(res==0)/res.shape[0],"Sub-process(es) done."))
    pool.close()
    pool.join()
