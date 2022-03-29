import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal as signal
from scipy.io import loadmat
'''
texturefilters = loadmat('D:\model_test/temper/数据/Face-anti-spoofing-based-on-color-texture-analysis-master/Features Extraction/texturefilters/ICAtextureFilters_7x7_8bit.mat')['ICAtextureFilters'][:]
im = cv.imread('23.jpg')
km = cv.cvtColor(im,cv.COLOR_BGR2RGB)
km = cv.resize(im,(360,360))
jm = cv.cvtColor(km,cv.COLOR_RGB2YCrCb)
lm = cv.cvtColor(km,cv.COLOR_RGB2HSV)
'''
def filter2(tem,imgwarp,mode):
    return signal.correlate2d(tem,imgwarp,mode)

def extractCode(img,texturefilters,mode='h'):

    # Initialization
    #img = double(img)
    end = img.shape[0]
    numScl = texturefilters.shape[2]
    codeBinary = np.zeros((img.shape[0],img.shape[1],numScl))

    # Wrap image
    r = np.floor(texturefilters.shape[0]/2.0).astype(int)
    upimg = img[:r,:]
    btimg = img[end-r:,:]
    lfimg = img[:,:r]
    rtimg = img[:,(end-r):]
    cr11 = img[:r,:r]
    cr12 = img[:r,end-r:]
    cr21 = img[end-r:,:r]
    cr22 = img[end-r:,end-r:]
    imgWrap = np.zeros((end+2*r,end+2*r))
    end = imgWrap.shape[0] 
    imgWrap[:r,:r] = cr22
    imgWrap[:r,r:end-r] = btimg
    imgWrap[:r,end-r:] = cr21

    imgWrap[r:end-r,:r] = rtimg
    imgWrap[end-r:,:r] = cr12
    
    imgWrap[r:end-r,end-r:] = lfimg
    imgWrap[end-r:,end-r:] = cr11

    imgWrap[end-r:,r:end-r] = upimg
    
    imgWrap[r:end-r,r:end-r] = img
    
    #for k in  range(len(imgWrap)):
    #    print(imgWrap[k].shape)
    # Loop over all kernels in a given set and calculate iris binary codes
    for i in range(numScl):
        tmp = texturefilters[:,:,i]
        ci = np.expand_dims(filter2(tmp,imgWrap,'valid'),2)
        # print(ci.shape)
        # pg = (ci>0)*2^(i-1)
        codeBinary = codeBinary + ((ci>0)*(2^(i))) ## +pg
    # print(codeBinary.shape)
    bsifscript = np.sum(codeBinary,axis=2,keepdims=True)
    # print(bsifscript.shape)
    #bsif,_ = np.histogram(bsifscript.flatten(),bins=2**numScl,range=(0,2**numScl))
    
    if mode=='h':
        bsif = np.bincount((bsifscript.flatten().astype('int')), minlength=256)
    else:
        bsif = (bsifscript.flatten() - np.mean(bsifscript.flatten())) / np.std(bsifscript.flatten())
    return bsif

# def extractCode(img,texturefilters):

#     # Initialization
#     #img = double(img)
#     end = img.shape[0]
#     numScl = texturefilters.shape[2]
#     codeBinary = np.zeros((img.shape[0],img.shape[1],numScl))

#     # Wrap image
#     r = math.floor(texturefilters.shape[0]/2)
#     upimg = img[:r,:]
#     btimg = img[end-r:,:]
#     lfimg = img[:,:r]
#     rtimg = img[:,(end-r):]
#     cr11 = img[:r,:r]
#     cr12 = img[:r,end-r:]
#     cr21 = img[end-r:,:r]
#     cr22 = img[end-r:,end-r:]
#     imgWrap = np.zeros((end+2*r,end+2*r))
#     end = imgWrap.shape[0] 
#     imgWrap[:r,:r] = cr22
#     imgWrap[:r,r:end-r] = btimg
#     imgWrap[:r,end-r:] = cr21

#     imgWrap[r:end-r,:r] = rtimg
#     imgWrap[end-r:,:r] = cr12
    
#     imgWrap[r:end-r,end-r:] = lfimg
#     imgWrap[end-r:,end-r:] = cr11

#     imgWrap[end-r:,r:end-r] = upimg
    
#     imgWrap[r:end-r,r:end-r] = img
    
#     #for k in  range(len(imgWrap)):
#     #    print(imgWrap[k].shape)
#     # Loop over all kernels in a given set and calculate iris binary codes
#     for i in range(numScl):
#         tmp = texturefilters[:,:,i]
#         ci = filter2(tmp,imgWrap,'valid')
#         #print(ci.shape)
#         codeBinary[:,:,i] = (ci>0)
#     return codeBinary
'''
codeBinary = extractCode(np.array(cv.cvtColor(cv.resize(im,(64,64)),cv.COLOR_BGR2GRAY)),texturefilters)
plt.plot(codeBinary.reshape(-1,1))
plt.show()
print(codeBinary.size)
'''