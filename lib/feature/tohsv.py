
import cv2 as cv
def convertRGBHSV(r, g, b): 
    # to change the range from 0..255 to 0..1: 
    r, g, b = r / 255.0, g / 255.0, b / 255.0
  
    # h, s, v = hue, saturation, value 
    cmax = max(r, g, b)    # maximum of r, g, b 
    cmin = min(r, g, b)    # minimum of r, g, b 
    diff = cmax-cmin       # diff of cmax and cmin. 

    if cmax == cmin:  
        h = 0
        
    elif cmax == r:  
        h = (60 * ((g - b) / diff) + 360) % 360

    elif cmax == g: 
        h = (60 * ((b - r) / diff) + 120) % 360

    elif cmax == b: 
        h = (60 * ((r - g) / diff) + 240) % 360

    if cmax == 0: 
        s = 0
    else: 
        s = (diff / cmax) * 100
  
    v = cmax * 100
    return h, s, v
im = cv.imread('23.jpg')
km = cv.cvtColor(im,cv.COLOR_BGR2RGB)
km = cv.resize(im,(64,64))
h,s,v = km[:,:,0],km[:,:,0],km[:,:,0]
for i in range(64):
    for j in range(64):
        h[i,j],s[i,j],v[i,j] = convertRGBHSV(km[i,j,0],km[i,j,1],km[i,j,2])