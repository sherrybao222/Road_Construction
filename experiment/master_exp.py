import pygame as pg
#import pygame_textinput
#import random
#import math
#from scipy.spatial import distance_matrix
#import numpy as np
import scipy.io as sio
from number_estimation import num_estimation
from road_basic import basic

# main
# =============================================================================
# setting up window, basic features 
pg.init()
pg.font.init()

# display setup
screen = pg.display.set_mode((2000, 1500), flags= pg.FULLSCREEN)  #  pg.FULLSCREEN pg.RESIZABLE
WHITE = (255, 255, 255)
screen.fill(WHITE)

# load maps
num_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test.mat',  struct_as_record=False)
n_trl_num = 2
basic_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_undo.mat',  struct_as_record=False)
n_trl_basic = 5

num_estimation(screen,num_map,n_trl_num)
basic(screen,basic_map,n_trl_basic)

n_trials = 5

pg.quit()

