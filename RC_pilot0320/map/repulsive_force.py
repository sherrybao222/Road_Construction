import scipy
import math
import scipy.optimize as optimize
import numpy as np

class circle_map:    
    def __init__(self):
        # map parameters
        self.N = 11     # total city number, including start
        self.radius = 10     # radius of city
        self.total = 400    # total budget
        self.budget_remain = 400    # remaining budget


        self.R = 400*400 #circle radius' sqaure
        self.r = np.random.uniform(0, self.R, self.N) 
        self.phi = np.random.uniform(0,2 * math.pi, self.N) 
        self.x = np.sqrt(self.r) * np.cos(self.phi) + 1000
        self.x = self.x.astype(int)
        self.y = np.sqrt(self.r) * np.sin(self.phi) + 950
        self.y = self.y.astype(int)
        self.xy = [(self.x[i], self.y[i]) for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city


def field(pos, city_x,city_y,sigma):
        
    return sum(scipy.exp(-((city_x - pos[0])**2+(city_y - pos[1])**2)/(2*sigma**2))/(2*math.pi*sigma**2))

mmap = circle_map()
def field_pos(mmap):
    position = []
    for i in range(0,11):
        initial_guess = [1, 1]
        cons = {'type': 'eq', 
            'fun': lambda pos: (pos[0] - mmap.x[i])**2 + (pos[1] - mmap.y[i])**2 - 30**2}
        result = optimize.minimize(field, initial_guess, 
                                   args=(mmap.x,mmap.y,30),constraints=cons)
        position.append(result.x)
    return position

pos = field_pos(mmap)