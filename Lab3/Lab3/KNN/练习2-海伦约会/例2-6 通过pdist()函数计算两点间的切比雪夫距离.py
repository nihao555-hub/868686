# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:09:46 2020

@author: Harry
"""

from scipy.spatial.distance import pdist
a = mat([1,2,3])
b = mat([4,5,6])
X = vstack([a,b])
print(pdist(X,'chebyshev'))
