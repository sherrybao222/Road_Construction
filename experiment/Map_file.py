import pygame as pg
import random
import math
from scipy.spatial import distance_matrix
from anytree import Node


# import numpy as np

# generate map and its corresponding parameters about people's choice
# -------------------------------------------------------------------------
class Map:
    def __init__(self):
        # map parameters
        self.N = 11  # total city number, including start
        # self.trial = 10 # number of trials
        self.radius = 7  # radius of city
        self.total = 700  # total budget
        self.budget_remain = 700  # remaining budget

        self.x = random.sample(range(500, 1400), self.N)  # x axis of all cities
        self.y = random.sample(range(500, 1400), self.N)  # y axis of all cities
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]  # combine x and y
        # self.trial_t = [[self.xy] for i in range(0, self.trial)] # combine trials together

        self.city_start = self.xy[0]  # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)

subject_1 = Map()
# print(subject_1.trial_t)
# print('Filename:', sub1, file=f)

trial = 10

for i in range(trial):
    with open('map_trial2.txt', "a") as f:
        i = Map()
        print(i.xy)
        print('Trial', i.xy, file=f)
