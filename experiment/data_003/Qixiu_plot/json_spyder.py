# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json

with open('/Users/fqx/Dropbox/Spring 2020/Ma Lab/GitHub/Road_Construction/experiment/data_003/test_all_3') as f:
  data = json.load(f)

for i in range(10):
    mapid = data[i]['mapid']
    mapid = int(sum(mapid)/len(mapid))
    #print(mapid)

total_city = 0
for i in range(20):
    n_city = data[i]['n_city'][-1]
    total_city += n_city

mean = total_city/20
#print(mean)

con1 = 0
con2 = 0
for i in range(144):
    if data[i]['cond'][-1]==2:
        n_city = data[i]['n_city'][-1]
        con1 += n_city
    if data[i]['cond'][-1]==3:
        n_city = data[i]['n_city'][-1]
        con2 += n_city

mean1 = con1/144
mean2 = con2/144
print(mean1)
print(mean2)