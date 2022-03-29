import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern as lbp  

def LBP(image, points=8, radius=1,mode='h'):
    '''
    Uniform Local Binary Patterns algorithm
    Input image with shape (height, width, channels)
    Output features with length * number of channels
    '''
    # calculate pattern length
    length = points**2 - abs(points - 3)
    # lbp per channel in image
    histogram = np.empty(0, dtype=np.int)
    # img_lbp = np.zeros((64, 64),np.uint8)
    for i in range(3):
        channel = image[:, :, i]
        pattern = lbp(channel, points, radius, method='nri_uniform')
        pattern = pattern.astype(np.int).ravel()
        # print(pattern.shape)
        pattern = np.bincount(pattern)
        # print(pattern.shape)
        if len(pattern) < length:
            pattern = np.concatenate((pattern, np.zeros(59 - len(pattern))))
            # print(pattern.shape)
        histogram = np.concatenate((histogram, pattern))
        # print(histogram.shape)
    # print(histogram.shape)
    # normalize the histogram and return it
    if mode=='h':
        return histogram
    else:
        features = (histogram - np.mean(histogram)) / np.std(histogram)
        return features
def LBP2(image, points=8, radius=1,mode='h'):
    '''
    Uniform Local Binary Patterns algorithm
    Input image with shape (height, width, channels)
    Output features with length * number of channels
    '''
    # calculate pattern length
    length = points**2 - abs(points - 3)
    # lbp per channel in image
    histogram = np.empty(0, dtype=np.int)
    img_lbp = np.zeros((64, 64),np.uint8)
    for i in range(3):
        channel = image[:, :, i]
        # pattern = lbp(channel, points, radius, method='nri_uniform')
        
        for i in range(1, 63):
            for j in range(1, 63):
                img_lbp[i, j] = lbp_calculated_pixel(channel, i, j)
        # print(img_lbp)
        pattern = img_lbp.astype(np.int).ravel()
        # print(pattern.shape)
        pattern = np.histogram(pattern,bins=58)[0]
        # print(pattern.shape)
        if len(pattern) < length:
            pattern = np.concatenate((pattern, np.zeros(59 - len(pattern))))
            # print(pattern.shape)
        histogram = np.concatenate((histogram, pattern))
        # print(histogram.shape)
    # print(histogram.shape)
    # normalize the histogram and return it
    if mode=='h':
        return histogram
    else:
        features = (histogram - np.mean(histogram)) / np.std(histogram)
        return features
    
def CoALBP(image, lbp_r=1, co_r=2, mode='h',image_norm=False):# 1 2 4  // 2 4 8
    '''
    Co-occurrence of Adjacent Local Binary Patterns algorithm 
    Input image with shape (height, width, channels)
    Input lbp_r is radius for adjacent local binary patterns
    Input co_r is radius for co-occurence of the patterns
    Output features with length 1024 * number of channels
    '''
    h, w, c = image.shape
    # normalize face
    if image_norm:
        image = (image - np.mean(image, axis=(0,1))) / (np.std(image, axis=(0,1)) + 1e-8)
    # albp and co-occurrence per channel in image
    histogram = np.empty(0, dtype=np.int)
    for i in range(image.shape[2]):
        C = image[lbp_r:h-lbp_r, lbp_r:w-lbp_r, i].astype(float)
        X = np.zeros((4, h-2*lbp_r, w-2*lbp_r))
        # adjacent local binary patterns
        X[0, :, :] = image[lbp_r  :h-lbp_r  , lbp_r+lbp_r:w-lbp_r+lbp_r, i] - C
        X[1, :, :] = image[lbp_r-lbp_r:h-lbp_r-lbp_r, lbp_r  :w-lbp_r  , i] - C
        X[2, :, :] = image[lbp_r  :h-lbp_r  , lbp_r-lbp_r:w-lbp_r-lbp_r, i] - C
        X[3, :, :] = image[lbp_r+lbp_r:h-lbp_r+lbp_r, lbp_r  :w-lbp_r  , i] - C
        X = (X>0).reshape(4, -1)
        # co-occurrence of the patterns
        A = np.dot(np.array([1, 2, 4, 8]), X)
        A = A.reshape(h-2*lbp_r, w-2*lbp_r) + 1
        hh, ww = A.shape
        D  = (A[co_r  :hh-co_r  , co_r  :ww-co_r  ] - 1) * 16 - 1
        Y1 =  A[co_r  :hh-co_r,   co_r+co_r:ww-co_r+co_r] + D
        Y2 =  A[co_r-co_r:hh-co_r-co_r, co_r+co_r:ww-co_r+co_r] + D
        Y3 =  A[co_r-co_r:hh-co_r-co_r, co_r  :ww-co_r  ] + D
        Y4 =  A[co_r-co_r:hh-co_r-co_r, co_r-co_r:ww-co_r-co_r] + D
        Y1 = np.bincount(Y1.ravel(), minlength=256)
        Y2 = np.bincount(Y2.ravel(), minlength=256)
        Y3 = np.bincount(Y3.ravel(), minlength=256)
        Y4 = np.bincount(Y4.ravel(), minlength=256)
        pattern = np.concatenate((Y1, Y2, Y3, Y4))
        histogram = np.concatenate((histogram, pattern))
    # normalize the histogram and return it
    if mode=='h':
        return histogram
    else:
        features = (histogram - np.mean(histogram)) / np.std(histogram)
        return features	
# local_binary_pattern 内部实现
def get_pixel(img, center, x, y):
	
	new_value = 0
	
	try:
		# If local neighbourhood pixel
		# value is greater than or equal
		# to center pixel values then
		# set it to 1
		if img[x][y] >= center:
			new_value = 1
			
	except:
		# Exception is required when
		# neighbourhood value of a center
		# pixel value is null i.e. values
		# present at boundaries.
		pass
	
	return new_value
# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):

	center = img[x][y]

	val_ar = []
	
	# top_left
	val_ar.append(get_pixel(img, center, x-1, y-1))
	
	# top
	val_ar.append(get_pixel(img, center, x-1, y))
	
	# top_right
	val_ar.append(get_pixel(img, center, x-1, y + 1))
	
	# right
	val_ar.append(get_pixel(img, center, x, y + 1))
	
	# bottom_right
	val_ar.append(get_pixel(img, center, x + 1, y + 1))
	
	# bottom
	val_ar.append(get_pixel(img, center, x + 1, y))
	
	# bottom_left
	val_ar.append(get_pixel(img, center, x + 1, y-1))
	
	# left
	val_ar.append(get_pixel(img, center, x, y-1))
	
	# Now, we need to convert binary
	# values to decimal
	power_val = [1, 2, 4, 8, 16, 32, 64, 128]

	val = 0
	
	for i in range(len(val_ar)):
		val += val_ar[i] * power_val[i]
		
	return val
'''
    path = '3.jpg'
    img_bgr = cv.imread(path,1)
    img_bgr = cv.resize(img_bgr,(64,64))

    height, width, _ = img_bgr.shape

    # We need to convert RGB image
    # into gray one because gray
    # image has one channel only.
    img_gray = cv.cvtColor(img_bgr,
                            cv.COLOR_BGR2GRAY)
    # Create a numpy array as
    # the same height and width
    # of RGB image
    img_lbp = np.zeros((height, width),
                    np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    hist_lbp = (img_lbp - np.mean(img_lbp)) / np.std(img_lbp)  # normalization
    #hist_lbp = cv.calcHist([img_lbp], [0], None, [256], [0,256])
    #plt.imshow(img_bgr)
    #plt.show()

    plt.imshow(img_lbp,cmap='gray')
    plt.show()

    print("LBP Program is finished")


    fea = LBP(img_bgr,8,1)
    print(fea.shape)
    plt.plot(fea)
    plt.show()

    fea = CoALBP(img_bgr,2,4)
    print(fea.shape)
    plt.plot(fea)
    plt.show()
'''