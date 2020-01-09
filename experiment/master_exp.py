import pygame as pg
import scipy.io as sio
from number_estimation import num_estimation
from road_basic import road_basic
from road_undo import road_undo

# main
# =============================================================================
# trial numbers
n_trl = [2,2,2]
n_all = 2 * sum(n_trl)
trials = [float("nan")] * n_all

n_1 = 1
n_2 = 1
n_3 = 1
# load maps
num_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test.mat',  struct_as_record=False)
basic_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic.mat',  struct_as_record=False)
undo_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_undo.mat',  struct_as_record=False)

# blocks
orders = [[1,2,3,3,2,1],
          [1,3,2,2,3,1],
          [2,1,3,3,1,2],
          [2,3,1,1,3,2],
          [3,1,2,2,1,3],
          [3,2,1,1,2,3]]

# information input
subject_num = input("Enter the subject number: ")
order_ind = int(input("Enter the order indicator: "))

# setting up window, basic features 
pg.init()
pg.font.init()

# display setup
screen = pg.display.set_mode((2000, 1600), flags= pg.FULLSCREEN)  #  pg.FULLSCREEN pg.RESIZABLE
WHITE = (255, 255, 255)
screen.fill(WHITE)

# blocks
start_trl = 0
for blk, cond in enumerate(orders[order_ind - 1]):   
    if cond == 1:
        trials[start_trl:] = num_estimation(screen,num_map,n_trl[cond-1],blk+1,n_1)
        n_1 = n_1 + 1
    if cond == 2:
        trials[start_trl:] = road_basic(screen,basic_map,n_trl[cond-1],blk+1,n_2)
        n_2 = n_2 + 1
    if cond == 3:    
        trials[start_trl:] = road_undo(screen,undo_map,n_trl[cond-1],blk+1,n_3)
        n_3 = n_3 + 1
    
    start_trl = start_trl + n_trl[cond-1]

pg.quit()

# saving mat
sio.savemat('test_all.mat', {'trials':trials})   

# saving csv
import csv
# =============================================================================

members = [attr for attr in dir(trials[0]) if not callable(getattr(trials[0], attr)) and not attr.startswith("__")]
exp = ['blk','cond','trl','mapid','time','pos','click','undo_press','choice_his','choice_loc',
              'budget_his','n_city','num_est']
info = ['N','R','phi','r','radius','total','x','y','xy','city_start','distance','order']

#exps = []
#infos = []
dict_exp = {}
dict_info = {}

for member in members:
    attr = [getattr(trial, member) for trial in trials]
    if member in set(exp):
        flat_list = [item for sublist in attr for item in sublist]
#        exps.append(flat_list)
        dict_exp[member] = flat_list        
    else:
        dict_info[member] = attr
#        infos.append(attr)
        
writefile = 'test_exp.csv'
with open( writefile, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(exp)
    all_values = [dict_exp[exp[i]] for i in range(len(exp))]
    writer.writerows(zip(*all_values))

writefile = 'test_info.csv'
with open( writefile, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(info)
    all_values = [dict_info[info[i]] for i in range(len(info))]
    writer.writerows(zip(*all_values))

