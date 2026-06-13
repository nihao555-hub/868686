# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

data = np.empty((200, 2))
data[:,0] = np.random.uniform(0., 100., size=200)
data[:,1] = 0.75 * data[:,0] + 1.5 + np.random.normal(0, 10., size=200)

plt.scatter(data[:,0], data[:,1])
plt.show()
