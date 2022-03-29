import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt


def lpq(img, winSize=7, freqestim=1, mode='h'):
    rho = 0.90  # if decorrelation  it is be used
    STFTalpha = 1 / winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    '''
    sigmaS = (winSize - 1) / 4  # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA = 8 / (winSize - 1)  # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)
    '''

    convmode = 'valid'  # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img = np.float64(img)  # Convert np.image to double
    r = (winSize - 1) / 2  # Get radius from window size
    x = np.arange(-r, r + 1)[np.newaxis]  # Form spatial coordinates in window

    if freqestim == 1:  # STFT uniform window
        #  Basic STFT filters
        w0 = np.ones_like(x)
        w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
        w2 = np.conj(w1)

    # Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)
    #print(filterResp1.shape)
    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                          filterResp2.real, filterResp2.imag,
                          filterResp3.real, filterResp3.imag,
                          filterResp4.real, filterResp4.imag])
    #print(freqResp.shape)
    # Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    #print(inds)
    LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)
    # print(LPQdesc)
    # Switch format to uint8 if LPQ code np.image is required as output
    if mode == 'im':
        LPQdesc = np.uint8(LPQdesc)
    # Histogram if needed
    if mode == 'h':
        LPQdesc = np.bincount(LPQdesc.flatten().astype('int'), minlength=256)
    else:
        LPQdesc = LPQdesc / LPQdesc.sum()
    # print(LPQdesc)
    return LPQdesc

'''
image = cv2.resize(cv2.imread('23.jpg'),(64,64))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img = ndimage.imread('image1.jpg')
# print(gray)
LPQdesc = lpq(gray)
#print(LPQdesc)
plt.plot(LPQdesc)
plt.show()
'''