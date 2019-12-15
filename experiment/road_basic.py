import pygame as pg
import random
import math
from scipy.spatial import distance_matrix
import numpy as np
import scipy.io as sio

# generate map and its corresponding parameters about people's choice
# ============================================================================
class Map:
    def __init__(self, map_content, trl_id, blk): 
        
#        self.circle_map()
        self.load_map(map_content, trl_id)
        self.data_init(blk, trl_id)
       
#   different maps
# ----------------------------------------------------------------------------
    def uniform_map(self):
        # map parameters
        self.N = 11     # total city number, including start
        self.radius = 7     # radius of city
        self.total = 700    # total budget
        self.budget_remain = 700    # remaining budget
        
        self.x = random.sample(range(500, 1400), self.N)    # x axis of all cities
        self.y = random.sample(range(500, 1400), self.N)    # y axis of all cities
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]   # combine x and y
   
        self.city_start = self.xy[0]    # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix
        
    def circle_map(self):
        # map parameters
        self.N = 1000     # total city number, including start
        self.radius = 10     # radius of city
        self.total = 400    # total budget
        self.budget_remain = 400    # remaining budget

        self.R = 400*400 #circle radius' sqaure
        self.r = np.random.uniform(0, self.R, self.N) 
        self.phi = np.random.uniform(0,2 * math.pi, self.N) 
        self.x = np.sqrt(self.r) * np.cos(self.phi) + 1000
        self.x = self.x.astype(int)
        self.y = np.sqrt(self.r) * np.sin(self.phi) + 800
        self.y = self.y.astype(int)
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix
      
    def load_map(self, map_content, trl_id):
        
        self.loadmap = map_content['map_list'][0,trl_id][0,0]
        self.order = np.nan
        
        self.N = self.loadmap.N.tolist()[0][0]
        self.radius = 10     # radius of city
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
    def make_choice(self, mouse):
        for i in range(1, self.N): # do not evaluate the starting point
            x2, y2 = mouse # mouse location
            self.mouse_distance = math.hypot(self.x[i] - x2, self.y[i] - y2)
            if (self.mouse_distance <= self.radius) and (i not in self.choice_dyn): # cannot choose what has been chosen
                self.index = i # index of chosen city
                self.city = self.xy[i] # location of chosen city
                self.check = 1 # indicator showing people made a valid choice
        
    def budget_update(self):
        dist = self.distance[self.index][self.choice_dyn[-1]] # get distance from current choice to previous choice
        self.budget_remain = self.budget_dyn[-1] - dist # budget update
       
    def check_end(self): # check if trial end
        distance_copy = self.distance[self.choice_dyn[-1]].copy() # copy distance list for current city
        for x in self.choice_dyn:
            distance_copy[x] = 0
        if any(i < self.budget_dyn[-1] and i != 0 for i in distance_copy):
            return True # not end
        else:
            return False # end
        
# -----------------------------------------------------------------------------           
    def data_init(self, blk, trl_id):
        self.blk = [blk]
        self.trl = [trl_id]
        self.cond = [2] # condition
        self.time = [round((pg.time.get_ticks()/1000), 2)] # mouse click time 
        self.pos = [pg.mouse.get_pos()]
        self.click = [0] # mouse click indicator
        self.undo_press = [0] # undo indicator
        
        self.choice_dyn = [0]
        self.choice_locdyn = [self.city_start]
        self.choice_his = [0]   # choice history, index
        self.choice_loc = [self.city_start] # choice location history
                
        self.budget_dyn = [self.total]
        self.budget_his = [self.total] # budget history

        self.n_city = [0] # number of cities connected
        self.check = 0 # indicator showing if people made a valid choice
        self.num_est = [np.nan] # number estimation input()
        
    def data(self, mouse, time, blk, trl_id): 
        self.blk.append(blk)
        self.trl.append(trl_id)
        self.cond.append(2)
        self.time.append(time)
        self.pos.append(mouse)
        self.click.append(1)
        self.undo_press.append(0)
        
        self.choice_dyn.append(self.index)
        self.choice_locdyn.append(self.city)
        self.choice_his.append(self.index)
        self.choice_loc.append(self.city)
                
        self.budget_dyn.append(self.budget_remain)
        self.budget_his.append(self.budget_remain)
                                
        self.n_city.append(self.n_city[-1] + 1)
        self.check = 0 # change choice indicator after saving them
        self.num_est.append(np.nan)

        del self.index, self.city   
        
    def static_data(self, mouse, time, blk, trl_id): 
        self.blk.append(blk)
        self.trl.append(trl_id)
        self.cond.append(2)
        self.time.append(time)
        self.pos.append(mouse)
        self.click.append(0)
        self.undo_press.append(0)
        
        self.choice_his.append(np.nan)
        self.choice_loc.append(np.nan)  
        self.budget_his.append(np.nan)
        
        self.n_city.append(self.n_city[-1])
        self.num_est.append(np.nan)
        
# visualize the game
# ============================================================================
class Draw: 
    def __init__(self, mmap,screen):
        self.instruction_submit(screen)
        self.cities(mmap,screen) # draw city dots
        if len(mmap.choice_dyn) >= 2: # if people have made choice, need to redraw the chosen path every time
            self.road(mmap,screen)
        self.text_write("Score: " + str(mmap.n_city[-1]), 100, BLACK, 1600, 200,screen) # show number of connected cities
         
    def road(self,mmap,screen): # if people have made choice, need to redraw the chosen path every time
        pg.draw.lines(screen, BLACK, False, mmap.choice_locdyn, 5)

    def cities(self,mmap,screen): # draw city dots       
        for city in mmap.xy[1:]: # exclude start city
            self.city = pg.draw.circle(screen, BLACK, city, mmap.radius)     
        self.start = pg.draw.circle(screen, RED, mmap.city_start, mmap.radius)
        
    def budget(self, mmap, mouse,screen):  
        # current mouse position
        cx, cy = mouse[0] - mmap.choice_locdyn[-1][0], mouse[1] - mmap.choice_locdyn[-1][1]
        # give budget line follow mouse in the correct direction
        radians = math.atan2(cy, cx)
        budget_pos = (int(mmap.choice_locdyn[-1][0] + mmap.budget_dyn[-1] * math.cos(radians)),
                      int(mmap.choice_locdyn[-1][1] + mmap.budget_dyn[-1] * math.sin(radians)))
        self.budget_line = pg.draw.line(screen, GREEN, mmap.choice_locdyn[-1], budget_pos, 5)

    def auto_snap(self, mmap,screen):
        pg.draw.line(screen, BLACK, mmap.choice_locdyn[-2], mmap.choice_locdyn[-1], 3)

    def instruction_submit(self,screen):
        self.text_write("Press Return to SUBMIT", 60, BLACK, 100, 200,screen)

    def game_end(self, mmap,screen): 
        self.text_write('Your score is ' + str(mmap.n_city[-1]), 100, BLACK, 700, 750,screen)
        self.text_write('Press Return to Next Trial ', 100, BLACK, 600, 850,screen)

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
def pygame_trial(all_done, trl_done, map_content, trl_id, screen, blk):
    
    trial = Map(map_content, trl_id, blk)
    pg.display.flip()
    screen.fill(WHITE)
    draw_map = Draw(trial,screen)
    
    while not trl_done:
        for event in pg.event.get():
            tick_second = round((pg.time.get_ticks()/1000), 2)
            mouse_loc = pg.mouse.get_pos()
            draw_map.budget(trial, mouse_loc,screen)
            
            if event.type == pg.QUIT:
                all_done = True
    
            elif event.type == pg.MOUSEMOTION:
                draw_map.budget(trial,mouse_loc,screen)
                trial.static_data(mouse_loc,tick_second,blk,trl_id)
           
            elif event.type == pg.MOUSEBUTTONDOWN:
                draw_map.budget(trial,mouse_loc,screen)
                trial.click[-1] = 1
                if trial.check_end(): # not end
                    trial.make_choice(mouse_loc)
                    if trial.check == 1: # made valid choice
                        trial.budget_update()
                        trial.data(mouse_loc, tick_second, blk, trl_id)
                        draw_map.auto_snap(trial,screen)                        
                    else:
                        trial.static_data(mouse_loc, tick_second, blk, trl_id)
                else: # end
                    print("The End") # need other end function
                
            elif event.type == pg.MOUSEBUTTONUP:
                draw_map.budget(trial,mouse_loc,screen)
                trial.static_data(mouse_loc,tick_second,blk,trl_id)
    
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    all_done = True   # very important, otherwise stuck in full screen
                    pg.quit()
                if event.key == pg.K_RETURN:
#                    pg.event.set_blocked(pg.MOUSEMOTION)
                    trl_done = True
    
            pg.display.flip()  
            screen.fill(WHITE)
            draw_map = Draw(trial,screen)
            
    
    while trl_done:
        for event in pg.event.get():
            screen.fill(WHITE)
            draw_map.game_end(trial,screen)
            pg.display.flip() 
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
                    trl_done = False 
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
    
    return all_done,trial,trl_done

def road_basic(screen,map_content,n_trials, blk):
     # conditions
    all_done = False
    trl_done = False
    
    trials = []
    
    # -------------------------------------------------------------------------
    while not all_done:
        for trl_id in range(0, n_trials):
            all_done,trial,trl_done = pygame_trial(all_done, trl_done, 
                                                   map_content, trl_id, screen, blk)
            trials.append(trial)
        all_done = True
    # saving
#    sio.savemat('test_saving_basic.mat', {'trials':trials}) 
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
    screen = pg.display.set_mode((2000, 1600), flags=pg.FULLSCREEN)  # pg.FULLSCREEN pg.RESIZABLE
 
    screen.fill(WHITE)
    
    # load maps
    map_content = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_undo.mat',  struct_as_record=False)
    n_trials = 5
    blk = 2 # set some number
 
    trials = road_basic(screen,map_content,n_trials,blk)
    
    pg.quit()
