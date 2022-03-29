import lib.feature.sid as sift
from lib.feature.lbp import LBP
from lib.feature.lbp import CoALBP
from lib.feature.lpq import lpq
from lib.feature.bsif import extractCode as bsif
import cv2 as cv
import numpy as np
def extract_faeture(im,texturefilters):
    # im_ycbcr = cv.cvtColor(im,cv.COLOR_BGR2YCrCb)
    # im_hsv = cv.cvtColor(im,cv.COLOR_BGR2HSV)
    # fea = LBP(im_ycbcr,8,1,'nh')
    # fea = np.concatenate((fea,LBP(im_hsv,8,1,'nh')))
    fea = np.empty(0)
    for j in  range(2):
        if j==0:    km = cv.cvtColor(im,cv.COLOR_BGR2YCrCb)
        elif j==1:    km = cv.cvtColor(im,cv.COLOR_BGR2HSV)
        fea = np.concatenate((fea,LBP(km,8,1,'nh')))
    cofea = np.empty(0)
    for j in  range(2):
        if j==0:    km = cv.cvtColor(im,cv.COLOR_BGR2YCrCb)
        elif j==1:    km = cv.cvtColor(im,cv.COLOR_BGR2HSV)
        for i in range(3):
            cofea = np.concatenate((cofea,CoALBP(km,2**i,2**(i+1),mode='h')[:]))
    # codeBinary = bsif(np.array(cv.cvtColor(im,cv.COLOR_BGR2GRAY)),texturefilters)
    codeBinary = np.empty(0)
    for j in  range(2):
        if j==0:    km = cv.cvtColor(im,cv.COLOR_BGR2YCrCb)
        elif j==1:    km = cv.cvtColor(im,cv.COLOR_BGR2HSV)
        for i in range(3):
            codeBinary = np.concatenate((codeBinary,bsif(np.array(km[:,:,i]),texturefilters)))
    LPQdesc = np.empty(0)
    for j in  range(2):
        if j==0:    km = cv.cvtColor(im,cv.COLOR_BGR2YCrCb)
        elif j==1:  km = cv.cvtColor(im,cv.COLOR_BGR2HSV)
        for i in range(3):
            LPQdesc = np.concatenate((LPQdesc,lpq(km[:,:,i])[:]))  
    fea = np.concatenate((fea,cofea))
    fea = np.concatenate((fea,LPQdesc))
    fea = np.concatenate((fea,codeBinary.flatten()))
    # print(fea.shape)
    # print(fea)
    # fea = scaler.transform(fea)
    fea = np.expand_dims(fea,0)
    return fea
