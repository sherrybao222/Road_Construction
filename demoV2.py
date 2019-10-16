import pygame as pg
import random
import numpy as np
import math
import scipy as sp
from scipy.spatial import distance_matrix

pg.init()
pg.font.init()
# conditions
done = False
# display setup
screen = pg.display.set_mode((1000, 650))   # display surface
clock = pg.time.Clock()
FPS = 30  # tracking time check how to use
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
city_size = 7
road_size = 3

screen.fill(WHITE)

# 后来我发现collision和click其实是一个东西，所以这个draft里面我就直接用collision
# 来检测是不是有点在城市上面，而不是随便乱点的
collision = []
click = [(0, 0)]
distance_his = [0]


class City:

    def __init__(self):
        self.x = random.randint(51, 649)  # set up in a way so don't overlap with budget information
        self.y = random.randint(51, 649)
        self.radius = 50
        self.xy = self.x, self.y

# 这个地方我用了distance matrix，但是不知道该怎么有效的在里面找数值
class Map:
    def __init__(self):
        self.city_start = City()
        self.cities = [City() for i in range(10)]
        self.all_city_xy = [city.xy for city in self.cities]
        self.all_city_xy.insert(0, self.city_start.xy)
        self.distance = sp.spatial.distance_matrix(self.all_city_xy, self.all_city_xy, p=2, threshold=10000)
        # don't really understand this p and threshold
        self.dis_history = [0]


class Collision:
    def __init__(self):
        self.coll_order = [0, 0]
        # second 0 = index of city_start in all_city.xy
        # self.coll_city = Collision().coll_check(pg.mouse.get_pos())
    # input: mouse_loc, output: selected city location
    @staticmethod
    def coll_check(mouse_loc):
        for city in Map().all_city_xy:
            x1, y1 = city[0], city[1]
            x2, y2 = mouse_loc  # pg.mouse.get_pos()
            distance = math.hypot(x1 - x2, y1 - y2)
            if distance < 14:  # radius for each city circle and mouse circle is 7
                return Map().all_city_xy.index(city)
                # x1, y1

# 这个部分就要求在distance matrix里面找数值，并用到updating budget 和budget check 里面
# budget check这个我觉得有点难，因为电脑要算出来你是不是game over 了
class Budget:
    def __init__(self):
        self.total = 700  # can change this total budget
        self.bud_history = [self.total]

    @staticmethod
    def budget_update(mouse_loc):   # input: mouse, output: new total budget
        previous_city = Collision().coll_order[-2]  # cities you have visited
        city1 = Map().all_city_xy.index(previous_city)  # the index of your previous city
        city2 = Collision().coll_check(mouse_loc)   # the index of your current city
        distance_search = Map().distance[city1, city2]  # find the pair-distance base on index
        budget_update = Budget().bud_history[-1] - distance_search
        return round(budget_update, 2)
        # when to append this new information to the list????

    @staticmethod   # given loc_a & distance for loc_b to draw line
    def budget_pos(city_x, city_y, d):  # d = budget remain bud_his[-1], xy = current city loc city[-1][0], [-1][1]
        cx, cy = pg.mouse.get_pos()[0] - click[-1][0], pg.mouse.get_pos()[1] - click[-1][1] # here use click, COLLISION?
        # current mouse position
        radians = math.atan2(cy, cx)
        return int(city_x + d * math.cos(radians)), int(city_y + d * math.sin(radians))

    @staticmethod   # double check still a bug
    def budget_check():
        city_visited = Collision().coll_order.pop(0)
        # all the cities visited, with starting city index of 0 == all_city.xy []
        current_budget = Budget().bud_history[-1]
        current_city = Collision().coll_order[-1]
        search_list = Map().distance[current_city, :]
        # this gives the entire columns of row of the current_city
        for dis in search_list:
            if dis not in city_visited:
                if all(Map().distance[current_city, dis]) > current_budget:
                    return False
                    # so all the pair-wise distance is outside of budget
                    # hence game-over
                else:
                    return True


map_1 = Map()
print(map_1.city_start.xy)
# print(map_1.all_city_xy)
print(map_1.distance)
print(map_1.distance[0, 1])

