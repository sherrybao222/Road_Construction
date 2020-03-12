import pygame as pg
import scipy.io as sio
from number_estimation import num_estimation
from road_basic import road_basic
from road_undo import road_undo
from training import training, incentive_instruction, end_instruction,payment
import json


# main
# =============================================================================
# trial numbers
n_trl = [2,2,2]
n_all = 2 * sum(n_trl)
trials = [float("nan")] * n_all

# map numbers
n_1 = 1
n_2 = 1
n_3 = 1

## load maps from mat
#train_num_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/training_test.mat',  struct_as_record=False)
#train_basic_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/training_basic_map.mat',  struct_as_record=False)
#train_undo_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/training_test_undo.mat',  struct_as_record=False)
#
#num_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test.mat',  struct_as_record=False)
#basic_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map_24.mat',  struct_as_record=False)
#undo_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_undo.mat',  struct_as_record=False)

# load maps from json
# /home/malab/Desktop/Road_Construction/
# /Users/sherrybao/Downloads/Research/Road_Construction/
# /Users/fqx/Dropbox/Spring 2020/Ma Lab/GitHub/Road_Construction/map/
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/num_training','r') as file: 
    train_num_map = json.load(file) 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_map_training','r') as file: 
    train_basic_map = json.load(file) 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/undo_map_training','r') as file: 
    train_undo_map = json.load(file) 

with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/num_48','r') as file: 
    num_map = json.load(file) 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map_48_all4','r') as file: 
    basic_map = json.load(file) 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/undo_map_48_all4','r') as file: 
    undo_map = json.load(file) 

# blocks
orders_1 = [[2,3,3,2],
          [3,2,2,3]]
orders_2 = [1,1]
# information input
subject_num = input("Enter the subject number: ")
order_ind = int(subject_num)%2
my_order_1 = orders_1[order_ind - 1]

# setting up window, basic features 
pg.init()
pg.font.init()

# display setup
WIDTH = 1900
HEIGHT = 1000
screen = pg.display.set_mode((WIDTH, HEIGHT), flags=pg.FULLSCREEN)  #  pg.FULLSCREEN pg.RESIZABLE
GREY = (222, 222, 222)

# -----------------------------------------------------------------------------
# training session exp 1
screen.fill(GREY)
training(screen)
mode_1 = 'try'
#num_estimation(screen,train_num_map,2,0,1,mode_1)
road_basic(screen,train_basic_map,2,0,1,mode_1)
road_undo(screen,train_undo_map,2,0,1,mode_1)

# incentive instruction exp 1
incentive_instruction(screen)

# game blocks exp 1
mode_2 = 'game'
start_trl = 0
for blk, cond in enumerate(my_order_1):   
#    if cond == 1:
#        trials[start_trl:] = num_estimation(screen,num_map,n_trl[cond-1],blk+1,n_1,mode_2)
#        n_1 = n_1 + 1
    if cond == 2:
        trials[start_trl:] = road_basic(screen,basic_map,n_trl[cond-1],blk+1,n_2,mode_2)
        n_2 = n_2 + 1
    if cond == 3:    
        trials[start_trl:] = road_undo(screen,undo_map,n_trl[cond-1],blk+1,n_3,mode_2)
        n_3 = n_3 + 1
    
    start_trl = start_trl + n_trl[cond-1]
    
# -----------------------------------------------------------------------------    
# training session exp 2
screen.fill(GREY)
num_estimation(screen,train_num_map,2,0,1,mode_1)

# game blocks exp 2
for blk, cond in enumerate(orders_2):   
    if cond == 1:
        trials[start_trl:] = num_estimation(screen,num_map,n_trl[cond-1],blk+5,n_1,mode_2)
        n_1 = n_1 + 1
    
    start_trl = start_trl + n_trl[cond-1]

# ----------------------------------------------------------------------------
# calculate payment
ind_list_2,pay_list_2 = payment(my_order_1,2,trials,1) 
ind_list_3,pay_list_3 = payment(my_order_1,3,trials,1)       
  
end_instruction(screen,ind_list_2,pay_list_2,ind_list_3,pay_list_3)

pg.quit()

# saving mat
sio.savemat('test_all_'+ subject_num + '.mat', {'trials':trials,'ind_list_2':ind_list_2,
                                                'pay_list_2':pay_list_2,'ind_list_3':ind_list_3,
                                                'pay_list_3':pay_list_3})   
# saving json
trial_json = [0]*len(trials)
for trl in range(len(trials)):
    trial_json[trl] = trials[trl].__dict__
with open('test_all_' + subject_num,'w') as file: 
    json.dump((trial_json,ind_list_2,pay_list_2,ind_list_3,pay_list_3),file)

# =============================================================================
# saving csv
import csv

members = [attr for attr in dir(trials[0]) if not callable(getattr(trials[0], attr)) and not attr.startswith("__")]
remove_members = ['budget_dyn','check','check_end_ind','choice_dyn','choice_locdyn','mouse_distance']
members = [ele for ele in members if ele not in remove_members] 
if 'position' in members: members.remove('position')
exp = ['blk','cond','trl','mapid','time','pos','click','undo_press','choice_his','choice_loc',
              'budget_his','n_city','num_est']
info = ['blk','cond','trl','mapid','N','R','phi','r','radius','total','x','y',
        'xy','city_start','distance','order']

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
    if member in set(info):
        if member in set(['blk','cond','trl','mapid']):
            attr = [x[0] for x in attr]
        dict_info[member] = attr
#        infos.append(attr)
        
with open('test_exp_' + subject_num + '.csv', 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(exp)
    all_values = [dict_exp[exp[i]] for i in range(len(exp))]
    writer.writerows(zip(*all_values))

with open('test_info_' + subject_num + '.csv', 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(info)
    all_values = [dict_info[info[i]] for i in range(len(info))]
    writer.writerows(zip(*all_values))
    writer.writerow(['ind_list_2','pay_list_2','ind_list_3','pay_list_3'])
    writer.writerow([ind_list_2,pay_list_2,ind_list_3,pay_list_3])

