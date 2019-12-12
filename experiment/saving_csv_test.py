#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:27:51 2019

@author: sherrybao
"""
# =============================================================================
# obj = trials[0]
# members = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]
# values = [getattr(obj, member) for member in members]
# 
# =============================================================================
# =============================================================================
# attrs = [o.attr for o in objs]
# 
# =============================================================================
import csv
import numpy as np

data1 = np.arange(10)
data2 = np.arange(10)*2
data3 = np.arange(10)*3

writefile = 'test.csv'
fieldnames = ['data1','data2', 'data3']
with open( writefile, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(zip(data1, data2, data3))
    
