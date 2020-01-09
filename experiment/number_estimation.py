import pygame as pg
import random
import math
from scipy.spatial import distance_matrix
import numpy as np
import scipy.io as sio

import pygame_textinput

# generate map and its corresponding parameters about people's choice
# =============================================================================
class Map:
    def __init__(self, map_content, trl_id, blk,map_id): 
        
        self.load_map(map_content, map_id)
        self.data_init(blk,trl_id,map_id)
        
#   different maps
# ----------------------------------------------------------------------------
    def uniform_map(self):
        # map parameters
        self.N = 11     # total city number, including start
        self.radius = 7     # radius of city
        self.total = 700    # total budget
        self.budget_remain = 700    # remaining budget()
        
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
        self.budget_remain = 700    # remaining budget()
        
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

    def load_map(self, map_content, map_id):
        
        self.loadmap = map_content['map_list'][0,map_id][0,0]
        self.order = map_content['order_list'][map_id]

        self.N = self.loadmap.N.tolist()[0][0]
        self.radius = 5     # radius of city
        self.total = self.loadmap.total.tolist()[0][0]   # total budget
        self.budget_remain = self.loadmap.total.copy().tolist()[0][0]  # remaining budget()

        self.R = self.loadmap.R.tolist()[0]
        self.r = self.loadmap.r.tolist()[0]
        self.phi = self.loadmap.phi.tolist()[0]
        self.x = self.loadmap.x.tolist()[0]
        self.y = self.loadmap.y.tolist()[0]
        self.xy = self.loadmap.xy.tolist()
        
        self.city_start = self.loadmap.city_start.tolist()[0]
        self.distance = self.loadmap.distance.tolist() 
        
# -----------------------------------------------------------------------------       
    def data_init(self, blk, trl_id, map_id):
        self.blk = [blk]
        self.trl = [trl_id]
        self.mapid = [map_id]
        self.cond = [1] # condition
        self.time = [round((pg.time.get_ticks()/1000), 2)] # mouse click time 
        self.pos = [pg.mouse.get_pos()]
        self.click = [0] # mouse click indicator
        self.undo_press = [0] # undo indicator
        
        self.choice_his = [np.nan]   # choice history, index
        self.choice_loc = [np.nan] # choice location history
                
        self.budget_his = [np.nan] # budget history

        self.n_city = [np.nan] # number of cities connected
        self.num_est = [np.nan] # number estimation input
        
    def data(self, mouse, time, text, blk, trl_id, map_id): 
        # history 
        self.blk.append(blk)
        self.trl.append(trl_id)
        self.mapid.append(map_id)
        self.cond.append(1)
        self.time.append(time)
        self.pos.append(mouse)
        self.click.append(0)
        self.undo_press.append(0)
        
        self.choice_his.append(np.nan) 
        self.choice_loc.append(np.nan)
                
        self.budget_his.append(self.budget_remain)
        
        self.n_city.append(np.nan)
        self.num_est.append(text)
                
# visualize the game
# =============================================================================
class Draw: 
    def __init__(self, mmap,screen):
        self.cities(mmap,screen) 
        self.city_order(mmap,screen)
        self.num_est(mmap,screen)
        
    def cities(self,mmap,screen): # draw city dots       
        for city in mmap.xy[1:]: # exclude start city
            self.city = pg.draw.circle(screen, BLACK, city, mmap.radius)     
        self.start = pg.draw.circle(screen, RED, mmap.city_start, mmap.radius)

    def city_order(self,mmap,screen):
        i = 1
        for order in mmap.order[1:]: # order of connection
            x = mmap.x[order] - 50
            y = mmap.y[order] 
            self.text_write(str(i), 50, BLACK, x, y,screen)
            i = i + 1
        
    def num_est(self, mmap,screen):
        self.text_write('How many cities can you connect? ', 60, BLACK, 100, 200,screen)
        self.text_write("Type your answer here: ", 60, BLACK, 100, 300,screen)
        self.text_write("Press Return to SUBMIT", 60, BLACK, 100, 400,screen)
        
    def budget(self, mmap, mouse,screen):  
        # current mouse position
        cx, cy = mouse[0] - mmap.city_start[0], mouse[1] - mmap.city_start[1]
        # give budget line follow mouse in the correct direction
        radians = math.atan2(cy, cx)
        budget_pos = (int(mmap.city_start[0] + mmap.total * math.cos(radians)),
                      int(mmap.city_start[1] + mmap.total * math.sin(radians)))
        self.budget_line = pg.draw.line(screen, GREEN, mmap.city_start, budget_pos, 4)

    def game_end(self, mmap,screen): 
        self.text_write('Press Return to Next Trial ', 100, BLACK, 600, 650,screen)
        
# helper function
# ----------------------------------------------------------------------------- 
    def text_write(self, text, size, color, x, y,screen):  # function that can display any text
        font_object = pg.font.SysFont(pg.font.get_default_font(), size)
        text_surface = font_object.render(text, True, color)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = x, y
        screen.blit(text_surface, text_rectangle.center)

# single trial
# =============================================================================
def pygame_trial(all_done, trl_done, map_content, trl_id, screen, blk, map_id):
    
    trial = Map(map_content, trl_id, blk, map_id)
    pg.display.flip()
    screen.fill(WHITE)
    draw_map = Draw(trial,screen)
    num_input = pygame_textinput.TextInput()
    
    while not trl_done:
        events = pg.event.get()
        for event in events:
            tick_second = round((pg.time.get_ticks()/1000), 2)
            mouse_loc = pg.mouse.get_pos()
            draw_map.budget(trial,mouse_loc,screen)
            
#            pg.event.set_blocked(pg.MOUSEBUTTONDOWN)
#            pg.event.set_blocked(pg.MOUSEBUTTONUP)

            # allow text-input on the screen
            num_input.update(events)
            screen.blit(num_input.get_surface(), (600, 300))
            # save estimation input
            text = num_input.get_text()
            if not text:
                text = np.nan
            trial.data(mouse_loc,tick_second,text,blk,trl_id,map_id)
            
            if event.type == pg.QUIT:
                all_done = True
    
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    all_done = True   # very important, otherwise stuck in full screen
                    pg.quit()
                if event.key == pg.K_RETURN and not np.isnan(float(trial.num_est[-1])):
                    trl_done = True
               
            pg.display.flip()
            screen.fill(WHITE)
            draw_map = Draw(trial,screen)

#    while trl_done:
#        events = pg.event.get()
#        for event in events:
#       
#            screen.fill(WHITE)
#            draw_map.game_end(trial,screen)
#            pg.display.flip()
#            
#            if event.type == pg.KEYDOWN:
#                if event.key == pg.K_RETURN:
#                    trl_done = False 
#            if event.type == pg.KEYDOWN:
#                if event.key == pg.K_ESCAPE:
#                    pg.quit()   
                 
    return all_done,trl_done,trial

def num_estimation(screen,map_content,n_trials,blk,n_blk):    
    # conditions
    all_done = False
    trl_done = False
    
    trials = []
    # running
    # -------------------------------------------------------------------------
    while not all_done:
        for trl_id in range(0, n_trials):
            map_id = trl_id + (n_blk - 1) * n_trials
            all_done,trl_done,trial = pygame_trial(all_done, trl_done, map_content, trl_id, screen, blk, map_id)
#            del trial.num_input # saving this variable will cause error
            trl_done = False 
            trials.append(trial)
        all_done = True
    # saving
#    sio.savemat('test_saving.mat', {'trials':trials})    
    return trials

# main
# =============================================================================
# setting up window, basic features 
    
WHITE = (255, 255, 255)
RED = (255, 102, 102)
GREEN = (0, 204, 102)
BLACK = (0, 0, 0)

if __name__ == "__main__":
    pg.init()
    pg.font.init()
    
    # display setup
    screen = pg.display.set_mode((2000, 1500), flags= pg.RESIZABLE)  #  pg.FULLSCREEN pg.RESIZABLE
    
    screen.fill(WHITE)
    
    # load maps
    map_content = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test.mat',  struct_as_record=False)
    n_trials = 2
    blk = 1 # set some number
    
    trials = num_estimation(screen,map_content,n_trials,blk)
    
    pg.quit()
