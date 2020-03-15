import pygame as pg
import pygame.gfxdraw
import random
import math
from scipy.spatial import distance_matrix
import numpy as np
import scipy.io as sio

import pygame_textinput

# generate map and its corresponding parameters about people's choice
# =============================================================================
class Map:
    def __init__(self, map_content, trl_id, blk, map_id): 
        
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
        
        self.loadmap = map_content[0][map_id]
        self.order = map_content[1][map_id]
        self.position = map_content[4][map_id]
        
        self.N = self.loadmap['N']
        self.radius = 5     # radius of city
        self.total = self.loadmap['total']   # total budget
        self.budget_remain = self.loadmap['total'] # remaining budget()
        
        self.R = self.loadmap['R']
        self.r = self.loadmap['r']
        self.phi = self.loadmap['phi']
        self.x = [x + int(WIDTH/2) for x in self.loadmap['x']] 
        self.y = [y + int(HEIGHT/2) for y in self.loadmap['y']]
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = self.loadmap['distance']
        
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
        self.budget(mmap, pg.mouse.get_pos(),screen)
        
        self.cities(mmap,screen) 
        self.city_order(mmap,screen)
        self.num_est(mmap,screen)
        
    def cities(self,mmap,screen): # draw city dots       
        for city in mmap.xy[1:]: # exclude start city
#            self.city = pg.draw.circle(screen, BLACK, city, mmap.radius)  
            self.city = draw_circle(screen,city,mmap.radius,BLACK)
#        self.start = pg.draw.circle(screen, RED, mmap.city_start, mmap.radius)
        self.start = draw_circle(screen,mmap.city_start,mmap.radius,RED)

    def city_order(self,mmap,screen):
        i = 1
        for order in mmap.order[1:]: # order of connection
            x = mmap.position[order][0] + int(WIDTH/2)
            y = mmap.position[order][1] + int(HEIGHT/2)
            text_write(str(i), 20, BLACK, x, y, screen)
#            pg.draw.circle(screen, RED, [int(x), int(y)], 4)
            i = i + 1
        
    def num_est(self, mmap,screen):
        text_write('How many cities could you connect? ', 60, BLACK, 100, 100,screen)
        text_write("Type your answer here: ", 60, BLACK, 100, 200,screen)
        text_write("Press RETURN to SUBMIT", 60, BLACK, 100, 300,screen)
        
    def budget(self, mmap, mouse,screen):  
        # current mouse position
        cx, cy = mouse[0] - mmap.city_start[0], mouse[1] - mmap.city_start[1]
        # give budget line follow mouse in the correct direction
        radians = math.atan2(cy, cx)
        budget_pos = [int(mmap.city_start[0] + mmap.total * math.cos(radians)),
                      int(mmap.city_start[1] + mmap.total * math.sin(radians))]
        self.budget_line = draw_line(screen,mmap.city_start, budget_pos, GREEN)
        
# instruction
# =============================================================================
def game_start(screen,blk): 
    text_write('This is Part '+ str(blk) + ' on Number Estimation',80, BLACK, 400, int(HEIGHT/3), screen)
    text_write('Estimate the number of cities you could connect with the given budget',
               50, BLACK, 400, int(HEIGHT/3)+200, screen)
    text_write('All cities must be connected in the labeled order', 50, BLACK, 400, int(HEIGHT/3)+300, screen)
    text_write('Press RETURN to continue.', 50, BLACK, 400, 900, screen)

def trial_start_1(screen):
    text_write('Now you will read the instruction for Number Estimation.', 50, BLACK, 50, 200, screen)
    text_write('In Number Estimation, you will see a map and a green line as your budget.', 50, BLACK, 50, 300, screen)
    text_write('You are asked to estimate the number of cities you could connect with the given budget.', 50, BLACK, 50, 400, screen)
    text_write('Please remember that all cities must be connected in the labeled order.', 50, BLACK, 50, 500, screen)
    text_write('You will type your response in a textbox.', 50, BLACK, 50, 600, screen)
    text_write('Press RETURN to see examples.', 50, BLACK, 50, 900, screen)

def trial_start_2(screen):
    text_write('You have finished the Road Construction with and without Undo.', 50, BLACK, 50, 200, screen)
    text_write('Please call the instructor to guide you through the instruction for Number Estimation.', 50, BLACK, 50, 300, screen)


def post_block(screen,blk):
    text_write('Congratulation, you finished Part '+ str(blk),100, BLACK, 400, int(HEIGHT/3), screen)
    text_write('You can take a short break now.',
               60, BLACK, 400, int(HEIGHT/3)+200, screen)
    text_write('Press RETURN to continue.', 60, BLACK, 400, 900, screen)
        
# helper function
# =============================================================================
def text_write(text, size, color, x, y,screen):  # function that can display any text
    font_object = pg.font.SysFont(pg.font.get_default_font(), size)
    text_surface = font_object.render(text, True, color)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = x, y
    screen.blit(text_surface, text_rectangle.center)

def draw_line(screen,X0, X1, color):
    mylist = [x + y for x, y in zip(X0, X1)]
    center_L1 = [x/2 for x in mylist]
    
    length = math.sqrt(((X0[0]-X1[0])**2)+((X0[1]-X1[1])**2)) # Line size
    thickness = 4
    angle = math.atan2(X0[1] - X1[1], X0[0] - X1[0])
    
    UL = (center_L1[0] + (length / 2.) * math.cos(angle) - (thickness / 2.) * math.sin(angle),
      center_L1[1] + (thickness / 2.) * math.cos(angle) + (length / 2.) * math.sin(angle))
    UR = (center_L1[0] - (length / 2.) * math.cos(angle) - (thickness / 2.) * math.sin(angle),
          center_L1[1] + (thickness / 2.) * math.cos(angle) - (length / 2.) * math.sin(angle))
    BL = (center_L1[0] + (length / 2.) * math.cos(angle) + (thickness / 2.) * math.sin(angle),
          center_L1[1] - (thickness / 2.) * math.cos(angle) + (length / 2.) * math.sin(angle))
    BR = (center_L1[0] - (length / 2.) * math.cos(angle) + (thickness / 2.) * math.sin(angle),
          center_L1[1] - (thickness / 2.) * math.cos(angle) - (length / 2.) * math.sin(angle))
    
    pygame.gfxdraw.aapolygon(screen, (UL, UR, BR, BL), color)
    pygame.gfxdraw.filled_polygon(screen, (UL, UR, BR, BL), color)

def draw_circle(screen,X,r,color):
    pygame.gfxdraw.aacircle(screen, X[0], X[1], r, color)
    pygame.gfxdraw.filled_circle(screen,X[0], X[1], r, color)

# single trial
# =============================================================================
def pygame_trial(all_done, trl_done, map_content, trl_id, screen, blk, map_id):
    
    trial = Map(map_content, trl_id, blk, map_id)
    screen.fill(GREY)
    draw_map = Draw(trial,screen)
    pg.display.flip()  
    num_input = pygame_textinput.TextInput()
    
    while not trl_done:
        events = pg.event.get()
        
        # allow text-input on the screen
        screen.fill(GREY)
        num_input.update(events)
        screen.blit(num_input.get_surface(), (600, 200)) # change cursor location
        draw_map = Draw(trial,screen)
        pg.display.flip()  
        
        # save estimation input
        text = num_input.get_text()

        for event in events:
            tick_second = round((pg.time.get_ticks()/1000), 2)
            mouse_loc = pg.mouse.get_pos()
#            draw_map.budget(trial,mouse_loc,screen)
            
            if not text:
                text = np.nan
            trial.data(mouse_loc,tick_second,text,blk,trl_id,map_id)
            
            if event.type == pg.QUIT:
                all_done = True
    
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    all_done = True   # very important, otherwise stuck in full screen
                    pg.display.quit()
                    pg.quit()
                try: 
                    float(trial.num_est[-1])
                    if event.key == pg.K_RETURN and not np.isnan(float(trial.num_est[-1])):
                        trl_done = True
                except: "Value error"
                
#    while trl_done:
#        events = pg.event.get()
#        for event in events:
#       
#            screen.fill(GREY)
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

def num_estimation(screen,map_content,n_trials,blk,n_blk,mode):    
    # conditions
    all_done = False
    trl_done = False
    
    trials = [] 
    
    if mode == 'game':
        screen.fill(GREY)
        game_start(screen,blk)
        pg.display.flip()
    elif mode == 'try':
        screen.fill(GREY)
        trial_start_1(screen)
        pg.display.flip()

    
    # instruction
    # -------------------------------------------------------------------------    
    ins = True
    while ins:
        events = pg.event.get()
        for event in events:
       
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
                    ins = False 
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.display.quit()
                    pg.quit()   

    if mode == 'try':
        screen.fill(GREY)
        trial_start_2(screen)
        pg.display.flip()

    ins = True
    while ins:
        events = pg.event.get()
        for event in events:
       
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
                    ins = False 
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.display.quit()
                    pg.quit()   

    # running
    # -------------------------------------------------------------------------
    while not all_done:
        for trl_id in range(0, n_trials):
#            if trl_id == 0:
#                Draw.game_end(screen)
            map_id = trl_id  + (n_blk - 1) * n_trials
            all_done,trl_done,trial = pygame_trial(all_done, trl_done, map_content, 
                                                   trl_id + 1, screen, blk, map_id)
#            del trial.num_input # saving this variable will cause error
            trl_done = False 
            trials.append(trial)
        all_done = True
    # saving
#    sio.savemat('test_saving.mat', {'trials':trials})  

    # end
    # -------------------------------------------------------------------------    
    if mode == 'game' and blk != 6:
        screen.fill(GREY)
        post_block(screen,blk)
        pg.display.flip()
    
        ins = True
        while ins:
            events = pg.event.get()
            for event in events:
           
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_RETURN:
                        ins = False 
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        pg.display.quit()
                        pg.quit()   
        
    
    return trials

# main
# =============================================================================
# setting up window, basic features 
    
GREY = (222, 222, 222)
RED = (255, 102, 102)
GREEN = (0, 204, 102)
BLACK = (0, 0, 0)

WIDTH = 1900
HEIGHT = 1000

if __name__ == "__main__":
    pg.init()
    pg.font.init()
    
    # display setup
    screen = pg.display.set_mode((WIDTH, HEIGHT))#, flags= pg.RESIZABLE)  #  pg.FULLSCREEN pg.RESIZABLE
    
    screen.fill(GREY)
    
    # load maps
#    map_content = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test.mat',  struct_as_record=False)
    import json
#    with open('/Users/fqx/Spring 2020/Ma Lab/GitHub/Road_Construction/map/num_48','r') as file:
    with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/num_48','r') as file: 
        map_content = json.load(file) 

    n_trials = 48
    blk = 1 # set some number\
    n_blk = 1
    mode = 'game'
    
    trials = num_estimation(screen,map_content,n_trials,blk,n_blk,mode)
    
    pg.display.quit()
    pg.quit()
