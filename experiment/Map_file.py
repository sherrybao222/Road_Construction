import pygame as pg
import random
import math
from scipy.spatial import distance_matrix
from anytree import Node
import csv


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

trial = 10

with open('trial.csv', 'w', newline='') as trial_file:
    trial_writer = csv.writer(trial_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    trial_writer.writerow(['trial_n']+['City_n'] + ['City_start'] + ['City_xy'] + ['Distance_m'] + ['Budget_t'] + ['Budget_r'])

for trial in range(trial):
    i = Map()
    trial_index = trial.__index__()
    with open('trial.csv', 'a', newline='') as trial_file:
        trial_writer = csv.writer(trial_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # trial_writer.writerow(['City_n'] + ['City_start'] + ['City_xy'] + ['Budget_t'] + ['Budget_r'])
        trial_writer.writerow([trial_index, i.N, i.city_start, i.xy, i.distance, i.total, i.budget_remain])

with open('trial.csv', newline='') as trial_file:
    trial_reader = csv.reader(trial_file, delimiter=' ', quotechar='|')
    for row in trial_reader:
        print(''.join(row))




