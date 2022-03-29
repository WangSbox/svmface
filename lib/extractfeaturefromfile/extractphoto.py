from __future__ import print_function
import os,time
import numpy as np
import  cv2 as cv
import multiprocessing
from .threadextract import zxcp as zxc
def photoextract(photopath,texturefilters,per_frame=4,num_worker=4,save=True,save_perframe=5000):
    r'''
    photopath:照片存放文件的上层路径 该路径下包含真或假的对应的文件夹
    texturefilters:
    per_frame:照片特征提取间隔数
    num_worker:进程池数量
    save:是否保存提取的特征,否则后面的save_perframe无效
    save_perframe:提取多少次特征后保存
    '''
    photo = os.listdir(photopath)
    print(photo)
    for i in range(len(photo)):
        begin = time.time()
        pool = multiprocessing.Pool(processes=num_worker)
        result = []
        photo_list = os.listdir(os.path.join(photopath,photo[i]))
        print(len(photo_list))
        count = 0
        for j in range(0,len(photo_list),per_frame):
            frame = cv.imread(os.path.join(os.path.join(photopath,photo[i]),photo_list[j]),1)
            if frame is  not None:
                result.append(pool.apply_async(zxc, (frame,texturefilters, )))
                count += 1
            if count % save_perframe ==0:
                pool.close()
                pool.join()
                data = []
                for re in result:
                    res = re.get()
                    if len(res)==21858:
                        # print(re.get())
                        data.append(res)
                data = np.array(data)
                print(data.shape)
                if save:
                    # if  os.path.exists(photo[i]) == False:
                    #     os.mkdir(photo[i])
                    np.save(photo[i] + str(int(count/save_perframe)) + '.npy',data)#单数组保存
                pool = multiprocessing.Pool(processes=num_worker)
                result = []
                # time.sleep(1000)
        pool.close()
        pool.join()
        end = time.time()
        print('path:{},Time cost:{:4f},{}'.format(photo[i],(end-begin)/60,"Sub-process(es) done."))
