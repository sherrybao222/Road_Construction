import pygame as pg
import pygame_textinput
import random
import math
from scipy.spatial import distance_matrix
import numpy as np
import scipy.io as sio


# generate map and its corresponding parameters about people's choice
# =============================================================================
class Map:
    def __init__(self, map_content, trl_id): 
        
        self.load_map(map_content, trl_id)
        self.num_input = pygame_textinput.TextInput()
        self.data_init()
        
#   different maps
# ----------------------------------------------------------------------------
    def uniform_map(self):
        # map parameters
        self.N = 11     # total city number, including start
        self.radius = 7     # radius of city
        self.total = 700    # total budget
        
        self.x = random.sample(range(500, 1400), self.N)    # x axis of all cities
        self.y = random.sample(range(500, 1400), self.N)    # y axis of all cities
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]   # combine x and y
   
        self.city_start = self.xy[0]    # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix
        
    def circle_map(self):
        # map parameters
        self.N = 11     # total city number, including start
        self.radius = 10     # radius of city
        self.total = 700    # total budget

        self.R = 450*450 #circle radius' sqaure
        self.r = np.random.uniform(0, self.R, self.N) 
        self.phi = np.random.uniform(0,2 * math.pi, self.N) 
        self.x = np.sqrt(self.r) * np.cos(self.phi) + 1000
        self.x = self.x.astype(int)
        self.y = np.sqrt(self.r) * np.sin(self.phi) + 950
        self.y = self.y.astype(int)
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix
    
    def load_map(self, map_content, trl_id):
        
        self.loadmap = map_content['map_list'][0,trl_id][0,0]
        self.order = map_content['order_list'][trl_id]

        self.radius = 10     # radius of city
        self.total = 700    # total budget
        
        self.R = self.loadmap.R
        self.r = self.loadmap.r
        self.phi = self.loadmap.phi
        self.x = self.loadmap.x
        self.y = self.loadmap.y
        self.xy = self.loadmap.xy
        
        self.city_start = self.loadmap.city_start.tolist()[0]
        self.distance = self.loadmap.distance 
        
# -----------------------------------------------------------------------------       
    def data_init(self):
        # history
        self.time = [round((pg.time.get_ticks()/1000), 2)] # mouse click time 
        self.pos = [pg.mouse.get_pos()]
        self.num_est = [] # number estimation input
    
    def static_data(self, mouse, time, text): 
        # history 
        self.time.append(time)
        self.pos.append(mouse)
        self.num_est.append(text)
        
# visualize the game
# =============================================================================
class Draw: 
    def __init__(self, mmap):
        self.cities(mmap) 
        self.city_order(mmap)
        self.num_est(mmap)
        
    def cities(self,mmap): # draw city dots       
        for city in mmap.xy[1:]: # exclude start city
            self.city = pg.draw.circle(screen, BLACK, city, 10)     
        self.start = pg.draw.circle(screen, RED, mmap.city_start, 10)

    def city_order(self,mmap):
        i = 1
        for order in mmap.order[1:]: # order of connection
            x = mmap.x[0,order] - 50
            y = mmap.y[0,order] 
            self.text_write(str(i), 50, BLACK, x, y)
            i = i + 1
        
    def num_est(self, mmap):
        self.text_write('How many cities can you connect? ', 60, BLACK, 100, 200)
        self.text_write("Type your answer here: ", 60, BLACK, 100, 300)
        self.text_write("Press Return to SUBMIT", 60, BLACK, 100, 400)
        
    def budget(self, mmap, mouse):  
        # current mouse position
        cx, cy = mouse[0] - mmap.city_start[0], mouse[1] - mmap.city_start[1]
        # give budget line follow mouse in the correct direction
        radians = math.atan2(cy, cx)
        budget_pos = (int(mmap.city_start[0] + mmap.total * math.cos(radians)),
                      int(mmap.city_start[1] + mmap.total * math.sin(radians)))
        self.budget_line = pg.draw.line(screen, GREEN, mmap.city_start, budget_pos, 5)

    def game_end(self, mmap): 
        self.text_write('Press Return to Next Trial ', 100, BLACK, 600, 650)

# helper function
# ----------------------------------------------------------------------------- 
    def text_write(self, text, size, color, x, y):  # function that can display any text
        font_object = pg.font.SysFont(pg.font.get_default_font(), size)
        text_surface = font_object.render(text, True, color)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = x, y
        screen.blit(text_surface, text_rectangle.center)

# single trial
# =============================================================================
def pygame_trial(all_done, trl_done, map_content, trl_id):
    
    trial = Map(map_content, trl_id)
    pg.display.flip()
    screen.fill(WHITE)
    draw_map = Draw(trial)
    
    while not trl_done:
        events = pg.event.get()
        for event in events:
            tick_second = round((pg.time.get_ticks()/1000), 2)
            mouse_loc = pg.mouse.get_pos()
            draw_map.budget(trial,mouse_loc)
            
            pg.event.set_blocked(pg.MOUSEBUTTONDOWN)
            pg.event.set_blocked(pg.MOUSEBUTTONUP)

            # allow text-input on the screen
            trial.num_input.update(events)
            screen.blit(trial.num_input.get_surface(), (600, 300))
    
            # save text-input
            text = trial.num_input.get_text()
            trial.static_data(mouse_loc,tick_second,text)
    
            if event.type == pg.QUIT:
                all_done = True
    
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    all_done = True   # very important, otherwise stuck in full screen
                    pg.display.quit()
                if event.key == pg.K_RETURN:
                    trl_done = True
               
            pg.display.flip()
            screen.fill(WHITE)
            draw_map = Draw(trial)
    
    while trl_done:
        events = pg.event.get()
        for event in events:
       
            screen.fill(WHITE)
            draw_map.game_end(trial)
            pg.display.flip()
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
                    trl_done = False 
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.display.quit()   
                 
    return all_done,trial,trl_done

# main
# =============================================================================
# setting up window, basic features 
pg.init()
pg.font.init()

# conditions
all_done = False
trl_done = False

# display setup
screen = pg.display.set_mode((2000, 1500), flags= pg.FULLSCREEN )  #  pg.FULLSCREEN pg.RESIZABLE
WHITE = (255, 255, 255)
RED = (255, 102, 102)
GREEN = (0, 204, 102)
BLACK = (0, 0, 0)
screen.fill(WHITE)

# load maps
map_content = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test.mat',  struct_as_record=False)
n_trial = 2

# -------------------------------------------------------------------------
while not all_done:
    for trl_id in range(0, n_trial):
        all_done,trial,trl_done = pygame_trial(all_done, trl_done, map_content, trl_id)
    all_done = True
while all_done:
    pg.display.quit()
    
# -------------------------------------------------------------------------
print("-----------------MAP INFORMATION --------------")
print("Starting City: " + str(trial.city_start))
print("city locations: " + str(trial.xy))
print("---------------- Break ----------------")
pg.quit()
