import scipy.io as sio
import numpy as np
import json
import math


class Map:
    def __init__(self, map_content, trl_id, degree=180):
        self.loadmap = map_content[trl_id]
        self.order = np.nan

        self.N = self.loadmap['N']
        self.total = self.loadmap['total']  # total budget
        self.budget_remain = self.loadmap['total']  # remaining budget()

        self.R = self.loadmap['R']
        self.r = self.loadmap['r']
        self.phi = [every + math.radians(degree) for every in self.loadmap['phi']]
        self.x = np.sqrt(self.r) * np.cos(self.phi)
        self.x = self.x.astype(int).tolist()
        self.y = np.sqrt(self.r) * np.sin(self.phi)
        self.y = self.y.astype(int).tolist()
        self.xy = [(self.x[i], self.y[i]) for i in range(0, len(self.x))]  # combine x and y

        self.city_start = self.xy[0]  # start city
        self.distance = self.loadmap['distance']


from glob import glob

basic_map = []
with open('./basicMap.json', 'rb') as file:
    basic_map_ = json.load(file)


# with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_map_training','r') as file:
#    basic_map = json.load(file)

degree_s = []
for degree in [90,180,270]:
    degree_s.extend((np.ones(int(len(basic_map_)/3)).astype(np.int16)* degree).tolist() )

degree_s = np.random.permutation(degree_s)
new_list = []
new_list_popLoadMap = []

for i in range(0, len(basic_map_)):  # [0]
    print('*' * 20)
    print(degree)
    degree  = degree_s[i]
    print('[' + str(int(i) + 1) + '/' + str(len(basic_map_)) + ']')
    map_ = Map(basic_map_, i,degree=degree)
    map_ = map_.__dict__
    new_list.append(map_)
    map_.pop('loadmap',None)
    new_list_popLoadMap.append(map_)
try:
    sio.savemat('undoMap_w_loadmap.mat', {'map_list': new_list},
                do_compression=True)
    sio.savemat('undoMap_wo_loadmap.mat', {'map_list': new_list_popLoadMap},
                do_compression=True)
except:
    ''
# saving json
with open('undoMap_w_loadmap.json', 'w') as file:
    json.dump(new_list, file)
with open('undoMap_wo_loadmap.json', 'w') as file:
    json.dump(new_list_popLoadMap, file)

print('UNDO')
## saving mat file
# sio.savemat('undo_map_training.mat', {'map_list':new_list})
## saving json
# with open('undo_map_training','w') as file:
#    json.dump(new_list,file)


