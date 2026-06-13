# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:07:44 2020

@author: Harry
"""

from numpy import*
from scipy.spatial.distance import pdist
v1 = mat([1,2])
v2 = mat([3,4])
x = vstack([v1,v2])
d = pdist(x)
print(d)
