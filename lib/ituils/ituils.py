from __future__ import print_function
 # do not delete this line if you want to save your log file.
import numpy as np
import  cv2 as cv
import pickle
import dlib # dlib for accurate face detection
import imutils # helper functions from pyimagesearch.com
from scipy.io import loadmat
import os
import time
import lib.feature.sid as sift
from lib.feature.lbp import LBP
from lib.feature.lbp import CoALBP
from lib.feature.lpq import lpq
from lib.feature.bsif import extractCode as bsif