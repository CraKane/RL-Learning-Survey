"""
@project = 20191119
@file = calcu
@author = 10374
@create_time = 2019/12/11 23:58
"""

import os
import math
import numpy as np
import pandas as pd
import scipy as sp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = 2*np.sin(12*np.pi/180)*2*np.sin(12*2*np.pi/180)*np.sin(12*3*np.pi/180)*2*np.sin(12*4*np.pi/180)*np.sin(12*5*np.pi/180)*2*np.sin(12*6*np.pi/180)*2*np.sin(12*7*np.pi)
b = a*a
print(b)
print(0.4/0.5)