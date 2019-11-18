#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:25:33 2019

@author: sherrybao
"""

mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance

import matplotlib.pyplot as plt
import numpy as np
x = np.random.multivariate_normal(mean, cov, 5000)
#plt.plot(x, y, 'x')
#plt.axis('equal')
#plt.show()