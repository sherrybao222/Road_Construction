import pygame as pg
#import pygame_textinput
#import random
#import math
#from scipy.spatial import distance_matrix
#import numpy as np
import scipy.io as sio
from number_estimation import num_estimation

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
map_content = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test.mat',  struct_as_record=False)
n_trials = 2

num_estimation(screen,map_content,n_trials)

pg.quit()

