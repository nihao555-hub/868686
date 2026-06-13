# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:08:44 2020

@author: Harry
"""

from numpy import*
from scipy.spatial.distance import pdist
m = mat([1,2,3])
n = mat([4,5,6])
X = vstack([m,n])
print(pdist(X,'cityblock'))
