#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:25:33 2019

@author: sherrybao
"""

mean = [0, 0]
cov = [[10000, 0], [0, 10000]]  # diagonal covariance

import matplotlib.pyplot as plt
import numpy as np
x,y = np.random.multivariate_normal(mean, cov, 10).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()