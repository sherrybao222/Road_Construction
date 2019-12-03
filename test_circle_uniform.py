#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:25:33 2019

@author: sherrybao
"""
import matplotlib.pyplot as plt
import numpy as np
from math import pi
N = 1000
r = np.random.uniform(2,4, N) 
phi = np.random.uniform(0,2*pi, N) 
x = np.sqrt(r) * np.cos(phi) 
y = np.sqrt(r) * np.sin(phi)
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()