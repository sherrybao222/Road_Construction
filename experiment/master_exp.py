import pygame as pg
#import pygame_textinput
#import random
#import math
#from scipy.spatial import distance_matrix
#import numpy as np
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

# load maps
num_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test.mat',  struct_as_record=False)
basic_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_undo.mat',  struct_as_record=False)
undo_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_undo.mat',  struct_as_record=False)

# blocks
orders = [[1,2,3,3,2,1],
          [1,3,2,2,3,1],
          [2,1,3,3,1,2],
          [2,3,1,1,3,2],
          [3,1,2,2,1,3],
          [3,2,1,1,2,3]]

# information input
subject_num = input("Enter the subject number: ")
gender = input("Enter the gender: ")
order_ind = int(input("Enter the order indicator: "))

# setting up window, basic features 
pg.init()
pg.font.init()

# display setup
screen = pg.display.set_mode((2000, 1500), flags= pg.FULLSCREEN)  #  pg.FULLSCREEN pg.RESIZABLE
WHITE = (255, 255, 255)
screen.fill(WHITE)

# blocks
start_trl = 0
for blk, cond in enumerate(orders[order_ind - 1]):   
    if cond == 1:
        trials[start_trl:] = num_estimation(screen,num_map,n_trl[cond-1],blk)
    if cond == 2:
        trials[start_trl:] = road_basic(screen,basic_map,n_trl[cond-1],blk)
    if cond == 3:    
        trials[start_trl:] = road_undo(screen,undo_map,n_trl[cond-1],blk)
    
    start_trl = start_trl + n_trl[cond-1]

# saving
sio.savemat('test_all.mat', {'trials':trials})   

pg.quit()

