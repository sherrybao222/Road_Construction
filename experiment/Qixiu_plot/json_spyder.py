# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

#loading pilot data
with open('/Users/fqx/Dropbox/Spring 2020/Ma Lab/GitHub/Road_Construction/experiment/data_002/test_all') as f:
  d1 = json.load(f).copy()
with open('/Users/fqx/Dropbox/Spring 2020/Ma Lab/GitHub/Road_Construction/experiment/data_003/test_all_3') as f:
  d2 = json.load(f).copy()
with open('/Users/fqx/Dropbox/Spring 2020/Ma Lab/GitHub/Road_Construction/test-room-experiment/data_009/test_all_9') as f:
  d3 = json.load(f).copy()

# list for operations
d1_rc = []
d2_rc = []
d3_rc = []

d1_undo = []
d2_undo = []
d3_undo = []

# d1_rc.clear()
# d2_rc.clear()
# d3_rc.clear()
#
# d1_undo.clear()
# d2_undo.clear()
# d3_undo.clear()

# score list for RC & undo condition
for i in range(144):
    if d1[i]['cond'][-1] == 2:
        d1_rc.append(d1[i]['n_city'][-1])
    if d2[i]['cond'][-1] == 2:
        d2_rc.append(d2[i]['n_city'][-1])
    if d3[i]['cond'][-1] == 2:
        d3_rc.append(d3[i]['n_city'][-1])
    if d1[i]['cond'][-1] == 3:
        d1_undo.append(d1[i]['n_city'][-1])
    if d2[i]['cond'][-1] == 3:
        d2_undo.append(d2[i]['n_city'][-1])
    if d3[i]['cond'][-1] == 3:
        d3_undo.append(d3[i]['n_city'][-1])

rc_con = [d1_rc[i] + d2_rc[i] + d3_rc[i] for i in range(len(d1_rc))]
undo_con = [d1_undo[i] + d2_undo[i] + d3_undo[i] for i in range(len(d1_undo))]

print(rc_con)
print(undo_con)

rc_ave = [round(rc_con[i]/3, 2) for i in range(len(rc_con))]
undo_ave = [round(undo_con[i]/3,2) for i in range(len(undo_con))]

print(rc_ave)
print(undo_ave)

# mapid list
mapid = []
for i in range(24):
    mapid.append(d1[i]['mapid'][-1])
print(mapid)

# line graph

# plot1 = plt.plot(mapid, rc_ave[:24], label='RC')
# plot2 = plt.plot(mapid, undo_ave[:24],label='Undo')
# plt.legend(loc="upper left")
# plt.xlabel('map number')
# plt.ylabel('cities connected')
# plt.show()


# bar graph

# data to plot
n_groups = len(mapid)
means_rc = rc_ave[:24]
means_undo = undo_ave[:24]

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_rc, bar_width,
alpha=opacity,
color='b',
label='RC')

rects2 = plt.bar(index + bar_width, means_undo, bar_width,
alpha=opacity,
color='g',
label='Undo')

plt.xlabel('Map Number')
plt.ylabel('Cities connected')
plt.title('Average cities connected across maps')
plt.xticks(index + bar_width, (i for i in range(len(mapid))))
plt.legend()

plt.tight_layout()
plt.show()


# practice
# for i in range(10):
#     mapid = d1[i]['mapid']
#     mapid = int(sum(mapid)/len(mapid))
#     #print(mapid)
#
# total_city = 0
# for i in range(20):
#     n_city = d1[i]['n_city'][-1]
#     total_city += n_city
# mean = total_city/20
#
# con1 = 0
# con2 = 0
# for i in range(144):
#     if d1[i]['cond'][-1]==2:
#         n_city = d1[i]['n_city'][-1]
#         con1 += n_city
#     if d1[i]['cond'][-1]==3:
#         n_city = d1[i]['n_city'][-1]
#         con2 += n_city
#
# mean1 = con1/144
# mean2 = con2/144
print(d1[0].keys())