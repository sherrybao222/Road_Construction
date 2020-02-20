import pygame as pg
import random
import math
from scipy.spatial import distance_matrix
import numpy as np
import scipy.io as sio

# generate map and its corresponding parameters about people's choice
# ============================================================================
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
        self.budget_remain = 700    # remaining budget
        
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
        self.budget_remain = 700    # remaining budget

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
        
        self.loadmap = map_content[map_id]
        self.order = np.nan
        
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
        distance_copy = self.distance[self.choice_dyn[-1]].copy()  # copy distance list for current city
        for x in self.choice_dyn:
            distance_copy[x] = 0
        if any(i < self.budget_dyn[-1] and i != 0 for i in distance_copy):
            return True # not end
        else:
            return False # end

# -----------------------------------------------------------------------------            
    def data_init(self,blk,trl_id,map_id):
        self.blk = [blk]
        self.trl = [trl_id]
        self.mapid = [map_id]
        self.cond = [3] # condition
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

        self.check_end_ind = 0
        
    def data(self, mouse, time, blk, trl_id, map_id): 
        self.blk.append(blk)
        self.trl.append(trl_id)
        self.mapid.append(map_id)
        self.cond.append(3)
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
        self.check = 0 # clear choice parameters after saving them
        self.num_est.append(np.nan)

        del self.index, self.city
    
    def undo(self, mouse, time, blk, trl_id, map_id):
        self.blk.append(blk)
        self.trl.append(trl_id)
        self.mapid.append(map_id)
        self.cond.append(3)
        self.time.append(time)
        self.pos.append(mouse)
        self.click.append(0)
        self.undo_press.append(1)
        
        self.choice_dyn.pop(-1)
        self.choice_locdyn.pop(-1)
        self.choice_his.append(self.choice_dyn[-1])
        self.choice_loc.append(self.choice_locdyn[-1])
        
        self.budget_dyn.pop(-1)                       
        self.budget_his.append(self.budget_dyn[-1])

        self.n_city.append(self.n_city[-1] - 1)
        self.num_est.append(np.nan)

    def static_data(self, mouse, time, blk, trl_id, map_id): 
        self.blk.append(blk)
        self.trl.append(trl_id)
        self.mapid.append(map_id)
        self.cond.append(3)
        self.time.append(time)
        self.pos.append(mouse)
        self.click.append(0)
        self.undo_press.append(0)
        
        self.choice_his.append(self.choice_dyn[-1])
        self.choice_loc.append(self.choice_locdyn[-1])  
        self.budget_his.append(self.budget_dyn[-1])
        
        self.n_city.append(self.n_city[-1])
        self.num_est.append(np.nan)

# score bar
# ============================================================================
class ScoreBar:
    def __init__(self,mmap):
        # only call once when initiated for this part
        # score bar parameters
        self.width = 100
        self.height = 500
        self.box = 12
        self.top = 200 # distance to screen top

        # center for labels
        self.box_center()
        # calculate incentive: N^2
        self.incentive()
        # incentive score indicator
        self.indicator(mmap)

    def box_center(self):
        self.box_height = self.height / self.box
        self.center_list = []
        self.uni_height = self.box_height / 2
        self.x = self.width / 2 + 1500 # larger the number, further to right

        for i in range(self.box):
            y =  i * self.box_height + self.uni_height
            loc = self.x, y
            self.center_list.append(loc)

    def incentive(self):
        self.score = list(range(0,self.box))
        self.incentive_score = []
        for i in self.score:
            i = i ** 2
            self.incentive_score.append(i)

    def indicator(self, mmap): # call this function to updates arrow location
        self.indicator_loc = self.center_list[mmap.n_city[-1]]
        self.indicator_loc_best = self.center_list[max(mmap.n_city)]
        
# visualize the game
# ============================================================================
class Draw: 
    def __init__(self, mmap,screen,scorebar):
        self.budget(mmap, pg.mouse.get_pos(),screen)

        self.instruction_undo(screen)
        self.cities(mmap,screen) # draw city dots
        self.number(scorebar, screen)
        self.white_arrow(scorebar, screen)
        self.arrow(scorebar,screen)
        
        if len(mmap.choice_dyn) >= 2: # if people have made choice, need to redraw the chosen path every time
            self.road(mmap,screen)
#        text_write("Score: " + str(mmap.n_city[-1]), 100, BLACK, WIDTH-200, 200,screen) # show number of connected cities

        if mmap.check_end_ind:
             self.check_end(screen)
         
    def road(self, mmap,screen): # if people have made choice, need to redraw the chosen path every time
        pg.draw.lines(screen, BLACK, False, mmap.choice_locdyn, 4)

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
        self.budget_line = pg.draw.line(screen, GREEN, mmap.choice_locdyn[-1], budget_pos, 4)

    def auto_snap(self, mmap,screen):
        pg.draw.line(screen, BLACK, mmap.choice_locdyn[-2], mmap.choice_locdyn[-1], 3)

    def instruction_undo(self,screen): 
        text_write("Press Z to UNDO", 30, BLACK, 100, 200,screen)
        text_write("Press SPACE to SUBMIT", 30, BLACK, 100, 300,screen)

    def check_end(self,screen):
        text_write("You are out of budget", 30, RED, 100, 400,screen)

# ---------------------------------------------------------------------------------
    def number(self, scorebar, screen):
        left = scorebar.center_list[0][0] - 25
        for i in range(scorebar.box):
            loc = scorebar.center_list[i]
            text = scorebar.incentive_score[i]
            text_write(str(text), int(scorebar.box_height - 15), BLACK, loc[0], loc[1]+scorebar.top , screen) # larger number, further to right
            pg.draw.rect(screen, BLACK, (left, loc[1]+scorebar.top-scorebar.uni_height, 
                                         scorebar.width, scorebar.box_height), 2)  # width for line thickness
    def arrow(self, scorebar,screen):
        # arrow parameter
        point = (scorebar.indicator_loc[0] - 30, scorebar.indicator_loc[1]+scorebar.top+10)
        v2 = point[0] - 20, point[1] + 20
        v3 = point[0] - 20, point[1] + 10
        v4 = point[0] - 40, point[1] + 10
        v5 = point[0] - 40, point[1] - 10
        v6 = point[0] - 20, point[1] - 10
        v7 = point[0] - 20, point[1] - 20
        self.vertices = [point, v2, v3, v4, v5, v6, v7]
        pg.draw.polygon(screen, BLACK, self.vertices)

    def white_arrow(self, scorebar,screen):
        # arrow parameter
        point = (scorebar.indicator_loc_best[0] - 30, scorebar.indicator_loc_best[1]+scorebar.top+10)
        v2 = point[0] - 20, point[1] + 20
        v3 = point[0] - 20, point[1] + 10
        v4 = point[0] - 40, point[1] + 10
        v5 = point[0] - 40, point[1] - 10
        v6 = point[0] - 20, point[1] - 10
        v7 = point[0] - 20, point[1] - 20
        self.vertices = [point, v2, v3, v4, v5, v6, v7]
        pg.draw.polygon(screen, WHITE, self.vertices)

# instruction
# =============================================================================
def game_start(screen): 
    text_write('Road Constructions with Undo', 100, BLACK, 300, int(HEIGHT/3), screen)
    text_write('Press Z to undo', 70, BLACK, 400, int(HEIGHT/3)+70, screen)
    text_write('Press SPACE to submit', 70, BLACK, 400, int(HEIGHT/3)+140, screen)
    
def trial_start(screen):
    text_write('This is Road Construction with Undo',50, BLACK, 50, 200, screen)
    text_write('The green line is your budget line, and you can rotate it by using your mouse.', 50, BLACK, 50, 300,screen)
    text_write('You are asked to connect as many cities as possible with the given budget.', 50, BLACK, 50, 400,screen)
    text_write('In addition, you can press Z to undo your connections.', 50, BLACK, 50, 500,screen)
    text_write('The score bar indicates your score in respect to the number of cities connected,', 50, BLACK, 50, 600,screen)
    text_write('and a record of your highest score achieved. ', 50, BLACK, 50, 700,screen)
    text_write('You will need to press Space to submit your final score. ', 50, BLACK, 50, 800,screen)
    text_write('Press SPACE to see examples', 50, BLACK, 50, 900, screen)

# helper function
# =============================================================================
def text_write(text, size, color, x, y,screen):  # function that can display any text
    font_object = pg.font.SysFont(pg.font.get_default_font(), size)
    text_surface = font_object.render(text, True, color)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = x, y
    screen.blit(text_surface, text_rectangle.center)

# single trial
# =============================================================================
def pygame_trial(all_done, trl_done, map_content, trl_id, screen, blk, map_id):
    
    trial = Map(map_content, trl_id, blk, map_id)
    scorebar = ScoreBar(trial)

    while not trl_done:
        
        screen.fill(GREY)
        draw_map = Draw(trial,screen,scorebar)
        pg.display.flip()  
        
        for event in pg.event.get():      
            tick_second = round((pg.time.get_ticks()/1000), 2)
            mouse_loc = pg.mouse.get_pos()
#            draw_map.budget(trial, mouse_loc, screen)
            
            if event.type == pg.QUIT:
                all_done = True
                
            elif event.type == pg.MOUSEMOTION:
#                draw_map.budget(trial,mouse_loc, screen)
                trial.static_data(mouse_loc,tick_second,blk,trl_id,map_id)
                
            elif event.type == pg.MOUSEBUTTONDOWN:
#                draw_map.budget(trial,mouse_loc, screen)
                if trial.check_end(): # not end
                    trial.make_choice(mouse_loc)
                    if trial.check == 1: # made valid choice
                        trial.budget_update()
                        trial.data(mouse_loc, tick_second,blk,trl_id,map_id)
                        draw_map.auto_snap(trial, screen)
                        scorebar.indicator(trial)
                    else:
                        trial.static_data(mouse_loc,tick_second,blk,trl_id,map_id)
                        trial.click[-1] = 1
                else: # end
                    print("The End") # need other end function
                
            elif event.type == pg.MOUSEBUTTONUP:
#                draw_map.budget(trial,mouse_loc, screen)
                trial.static_data(mouse_loc,tick_second,blk,trl_id,map_id)
                if  not trial.check_end(): 
                    trial.check_end_ind = 1

            elif event.type == pg.KEYDOWN:
                if (pg.key.get_pressed() and trial.choice_dyn[-1] != 0
                    and event.key == pg.K_z):
                    trial.undo(mouse_loc, tick_second, blk,trl_id,map_id)
                    scorebar.indicator(trial)
#                    print("budget undo" + str(trial.budget_dyn))
                    
                if event.key == pg.K_ESCAPE:    
                    all_done = True   # very important, otherwise stuck in full screen
                    pg.display.quit()
                    pg.quit()
                if event.key == pg.K_SPACE and trial.n_city[-1] != 0:
#                    pg.event.set_blocked(pg.MOUSEMOTION)
                    trl_done = True
    
            elif event.type == pg.KEYUP:
                if event.key == pg.K_z:
                    if  trial.check_end(): 
                        trial.check_end_ind = 0
    
#    while trl_done:
#        for event in pg.event.get():
#            
#            screen.fill(GREY)
#            draw_map.game_end(trial, screen)
#            pg.display.flip() 
#            
#            if event.type == pg.KEYDOWN:
#                if event.key == pg.K_RETURN:
#                    trl_done = False 
#            if event.type == pg.KEYDOWN:
#                if event.key == pg.K_ESCAPE:
#                    pg.quit()

    return all_done,trial,trl_done

def road_undo(screen,map_content,n_trials,blk,n_blk,mode):
    # conditions
    all_done = False
    trl_done = False
    
    trials = []

    if mode == 'game':
        screen.fill(GREY)
        game_start(screen)
        pg.display.flip()
    elif mode == 'try':
        screen.fill(GREY)
        trial_start(screen)
        pg.display.flip()

    # instruction
    # -------------------------------------------------------------------------    
    ins = True
    while ins:
        events = pg.event.get()
        for event in events:
                   
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    ins = False 
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.display.quit()
                    pg.quit()   

    # running        
    # -------------------------------------------------------------------------
    while not all_done:
        for trl_id in range(0, n_trials):
            map_id = trl_id + (n_blk - 1) * n_trials
            all_done,trial,trl_done = pygame_trial(all_done, trl_done, map_content, 
                                                   trl_id + 1, screen,blk,map_id)
            trl_done = False
            trials.append(trial)
        all_done = True
    #while all_done:
    #    pg.display.quit()
        
    # saving
#    sio.savemat('test_saving_undo.mat', {'trials':trials})   
    return trials

# main
# =============================================================================
# setting up window, basic features 

WHITE = (255, 255, 255)    
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
    screen = pg.display.set_mode((WIDTH, HEIGHT), flags=pg.FULLSCREEN)  # pg.FULLSCREEN pg.RESIZABLE

    screen.fill(GREY)
    
    # load maps
    import json
    with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/undo_map_24','r') as file: 
        map_content = json.load(file) 

    n_trials = 5
    blk = 3 # set to some number
    n_blk = 1
    mode = 'game'
    
    trials = road_undo(screen,map_content,n_trials,blk,n_blk,mode)

    pg.display.quit()
    pg.quit()
