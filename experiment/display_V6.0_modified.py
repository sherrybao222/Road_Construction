import pygame as pg
import random
import math
from scipy.spatial import distance_matrix
from anytree import Node
import numpy as np

# generate map and its corresponding parameters about people's choice
# -------------------------------------------------------------------------
class Map:
    def __init__(self): 
        # map parameters
        self.N = 11 # total city number, including start
        self.radius = 7 # radius of city
        self.total = 700 # total budget
        self.budget_remain = 700 # remaining budget
        
        mean = [350, 350]
        cov = [[20000, 0], [0, 20000]]  # diagonal covariance
        
        self.xy = np.random.multivariate_normal(mean, cov, self.N)
        self.xy = self.xy.astype(int)
        self.x, self.y = self.xy.T #transpose
        
        #self.x = random.sample(range(51, 649), self.N) # x axis of all cities
        #self.y = random.sample(range(51, 649), self.N) # y axis of all cities
        #self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))] # combine x and y
   
        self.city_start = self.xy[0] # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000) # city distance matrix
        
        # people's decisions
        self.choice = Node(0, budget = self.total) # start choice in tree
        self.click = [] # mouse click location
        self.click_time = [] # mouse click time 
        self.budget_his = [self.total] # budget history
        self.choice_his = [0] # choice history
        self.choice_loc = [self.city_start] # choice location history
        self.n_city = 0 # number of cities connected
        self.check = 0 # indicator showing if people made a valid choice
    
    def make_choice(self, mouse):
        for i in range(1, self.N): # do not evaluate the starting point
            x2, y2 = mouse # mouse location
            self.mouse_distance = math.hypot(self.x[i] - x2, self.y[i] - y2)
            if (self.mouse_distance <= self.radius) and (i not in self.choice_his): # cannot choose what has been chosen
                self.index = i # index of chosen city
                self.city = self.xy[i] # location of chosen city
                self.check = 1 # indicator showing people made a valid choice
        
    def budget_update(self):
        dist =  self.distance[self.index][self.choice_his[-1]] # get distance from current choice to previous choice
        self.budget_remain = self.budget_remain - dist # budget update
       
    def data(self, mouse): 
        tick_second = round((pg.time.get_ticks()/1000), 2)
        self.click_time.append(tick_second)
        self.click.append(mouse)
       
        self.budget_his.append(self.budget_remain)
        self.choice_his.append(self.index)  #which index is this referring to? 
        self.choice_loc.append(self.city)
        new = Node(self.index, parent = self.choice, budget = self.budget_remain, time = tick_second)
        
        self.n_city = self.n_city + 1
        self.choice = new
        
        self.check = 0 # clear choice parameters after saving them
        del self.index, self.city
        
    def check_end(self): # check if trial end
        distance_copy = self.distance[self.choice_his[-1]] # copy distance list for current city
        if any(i < self.budget_his[-1] and i != 0 for i in distance_copy):
            return True # not end
        else:
            return False # end
        
    def undo(self): # undo function, haven't tested
        new = self.choice.parent
        self.choice = new
        budget = self.budget_his[-2]
        self.budget_his.append(budget)
        choice = self.choice_his[-2]
        self.choice_his.append(choice)
        
# visualize the game
# -------------------------------------------------------------------------
class Draw: 
    def __init__(self, mmap):
        self.cities(mmap) # draw city dots
        if len(mmap.choice_his) >= 2: # if people have made choice, need to redraw the chosen path every time
            self.road(mmap)
        self.text_write(str(mmap.n_city), 100, BLACK, 900, 100) # show number of connected cities
         
    def road(self,mmap): # if people have made choice, need to redraw the chosen path every time
        pg.draw.lines(screen, BLACK, False, mmap.choice_loc, 3)

    def cities(self,mmap): # draw city dots       
        for city in mmap.xy[1:]: # exclude start city
            self.city = pg.draw.circle(screen, BLACK, city, 7)     
        self.start = pg.draw.circle(screen, RED, mmap.city_start, 7)
        
    def budget(self, mmap, mouse):  
        # current mouse position
        cx, cy = mouse[0] - mmap.choice_loc[-1][0], mouse[1] - mmap.choice_loc[-1][1]
        # give budget line follow mouse in the correct direction
        radians = math.atan2(cy, cx)
        budget_pos = (int(mmap.choice_loc[-1][0] + mmap.budget_his[-1] * math.cos(radians)), 
                      int(mmap.choice_loc[-1][1] + mmap.budget_his[-1] * math.sin(radians)))

        self.budget_line = pg.draw.line(screen, GREEN, mmap.choice_loc[-1], budget_pos, 3)
        
    def auto_snap(self, mmap):
        pg.draw.line(screen, BLACK, mmap.choice_loc[-2], mmap.choice_loc[-1], 3)

    def text_write(self, text, size, color, x, y):  # function that can display any text
        font_object = pg.font.SysFont(pg.font.get_default_font(), size)
        text_surface = font_object.render(text, True, color)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = x, y
        screen.blit(text_surface, text_rectangle.center)


#------------------------------------------------------------------------------
# setting up window, basic features 
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
screen.fill(WHITE)
clock.tick(FPS)
# -------------------------------------------------------------------------
trial = Map()
draw_map = Draw(trial) 
# -------------------------------------------------------------------------
# loop for displaying until quit
while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
            
        if event.type == pg.MOUSEMOTION:
            draw_map.budget(trial,pg.mouse.get_pos())
            
        if event.type == pg.MOUSEBUTTONDOWN:
            mouse_loc = pg.mouse.get_pos()
            draw_map.budget(trial,mouse_loc)
            if trial.check_end(): # not end
                trial.make_choice(mouse_loc)
                if trial.check == 1: # made valid choice
                    trial.budget_update()
                    trial.data(mouse_loc)
                    draw_map.auto_snap(trial)
            else: # end
                print("The End") # need other end function
            
        if event.type == pg.MOUSEBUTTONUP: 
            draw_map.budget(trial,pg.mouse.get_pos())
            
        pg.display.flip()  
        screen.fill(WHITE)
        draw_map = Draw(trial)


# -------------------------------------------------------------------------
print("-----------------MAP INFORMATION --------------")
print("Starting City: " + str(trial.city_start))
print("city locations: " + str(trial.xy))
print("---------------- Break ----------------")
pg.quit()
