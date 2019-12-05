import pygame as pg
import random
import math
from scipy.spatial import distance_matrix
import numpy as np
import scipy.io as sio

# generate map and its corresponding parameters about people's choice
# ============================================================================
class Map:
    def __init__(self, map_content): 
        
        self.load_map(map_content)
        self.data_init()

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
    
        self.n_city = 0 # number of cities connected
        self.check = 0 # indicator showing if people made a valid choice
    
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
    
        self.n_city = 0 # number of cities connected
        self.check = 0 # indicator showing if people made a valid choice
    
    def load_map(self, map_content):
        
        self.loadmap = map_content['map_list'][0,0][0,0]

        self.N = self.loadmap.N.tolist()[0][0]
        self.radius = 10     # radius of city
        self.total = self.loadmap.total   # total budget
        self.budget_remain = 700    # remaining budget
        
        self.R = self.loadmap.R
        self.r = self.loadmap.r
        self.phi = self.loadmap.phi
        self.x = self.loadmap.x.tolist()[0]
        self.y = self.loadmap.y.tolist()[0]
        self.xy = self.loadmap.xy.tolist()
        
        self.city_start = self.loadmap.city_start.tolist()[0]
        self.distance = self.loadmap.distance.tolist()
        
        self.n_city = 0 # number of cities connected
        self.check = 0 # indicator showing if people made a valid choice

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
        distance_copy = self.distance[self.choice_dyn[-1]]  # copy distance list for current city
        if any(i < self.budget_dyn[-1] and i != 0 for i in distance_copy):
            return True # not end
        else:
            return False # end
        
    def undo(self, mouse, time):
        # dynamic change
        self.choice_dyn.pop(-1)
        self.choice_locdyn.pop(-1)
        self.budget_dyn.pop(-1)
        self.n_city = self.n_city - 1
        
        # save history
        self.time.append(time)
        self.pos.append(mouse)
        
        self.budget_his.append(self.budget_dyn[-1])
        self.choice_his.append(self.choice_dyn[-1])
        self.choice_loc.append(self.choice_locdyn[-1])
        
        self.click.append(0)
        self.undo_press.append(1)
# -----------------------------------------------------------------------------          
    def data_init(self):
        # dynamic 
        self.choice_dyn = [0]
        self.choice_locdyn = [self.city_start]
        self.budget_dyn = [self.total]

        # history
        self.time = [0] # mouse click time 
        self.pos = [0]
        
        self.choice_his = [0]   # choice history, index
        self.choice_loc = [self.city_start] # choice location history
        
        self.click = [0] # mouse click indicator
        self.undo_press = [0] # undo indicator
        
        self.budget_his = [self.total] # budget history
        
    def data(self, mouse, time): 
        # dynamic
        self.choice_dyn.append(self.index)
        self.choice_locdyn.append(self.city) 
        self.budget_dyn.append(self.budget_remain)
        
        # history     
        self.time.append(time)
        self.pos.append(mouse)
        
        self.choice_his.append(self.index) 
        self.choice_loc.append(self.city)
        
        self.click.append(1)
        self.undo_press.append(0)

        self.budget_his.append(self.budget_remain)
   
        self.n_city = self.n_city + 1
        
        self.check = 0 # clear choice parameters after saving them
        del self.index, self.city
    
    def static_data(self, mouse, time): 
        # history 
        self.time.append(time)
        self.pos.append(mouse)
        
        self.choice_his.append(float('nan'))
        self.choice_loc.append(float('nan'))  
        
        self.budget_his.append(float('nan'))
        
        self.click.append(0)
        self.undo_press.append(0)
        
# visualize the game
# ============================================================================
class Draw: 
    def __init__(self, mmap):
        self.instruction_undo()
        self.cities(mmap) # draw city dots
        if len(mmap.choice_dyn) >= 2: # if people have made choice, need to redraw the chosen path every time
            self.road(mmap)
        self.text_write("Score: " + str(mmap.n_city), 100, BLACK, 1600, 200) # show number of connected cities
         
    def road(self, mmap): # if people have made choice, need to redraw the chosen path every time
        pg.draw.lines(screen, BLACK, False, mmap.choice_locdyn, 5)

    def cities(self,mmap): # draw city dots       
        for city in mmap.xy[1:]: # exclude start city
            self.city = pg.draw.circle(screen, BLACK, city, mmap.radius)     
        self.start = pg.draw.circle(screen, RED, mmap.city_start, mmap.radius)
        
    def budget(self, mmap, mouse):  
        # current mouse position
        cx, cy = mouse[0] - mmap.choice_locdyn[-1][0], mouse[1] - mmap.choice_locdyn[-1][1]
        # give budget line follow mouse in the correct direction
        radians = math.atan2(cy, cx)
        budget_pos = (int(mmap.choice_locdyn[-1][0] + mmap.budget_dyn[-1] * math.cos(radians)),
                      int(mmap.choice_locdyn[-1][1] + mmap.budget_dyn[-1] * math.sin(radians)))
        self.budget_line = pg.draw.line(screen, GREEN, mmap.choice_locdyn[-1], budget_pos, 5)

    def auto_snap(self, mmap):
        pg.draw.line(screen, BLACK, mmap.choice_locdyn[-2], mmap.choice_locdyn[-1], 3)

    def instruction_undo(self): 
        self.text_write("Press Z to UNDO", 60, BLACK, 100, 200)
        self.text_write("Press Return to SUBMIT", 60, BLACK, 100, 300)

    def game_end(self, mmap): 
        #pg.draw.rect(screen, WHITE, (600, 600, 600, 200), 0)
        self.text_write('Your score is ' + str(mmap.n_city), 100, BLACK, 700, 750)
        pg.display.update()
        
# helper function
# ----------------------------------------------------------------------------- 
    def text_write(self, text, size, color, x, y):  # function that can display any text
        font_object = pg.font.SysFont(pg.font.get_default_font(), size)
        text_surface = font_object.render(text, True, color)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = x, y
        screen.blit(text_surface, text_rectangle.center)


# main
# =============================================================================
# setting up window, basic features 
pg.init()
pg.font.init()

# conditions
done = False

# display setup
screen = pg.display.set_mode((2000, 1500), flags=pg.FULLSCREEN)  # display surface
WHITE = (255, 255, 255)
RED = (255, 102, 102)
GREEN = (0, 204, 102)
BLACK = (0, 0, 0)
screen.fill(WHITE)

# load maps
map_content = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_undo.mat',  struct_as_record=False)
n_trial = 5
# -------------------------------------------------------------------------
trial = Map(map_content)
draw_map = Draw(trial)
# -------------------------------------------------------------------------
# loop for displaying until quit
while not done:
    for event in pg.event.get():      
        tick_second = round((pg.time.get_ticks()/1000), 2)
        mouse_loc = pg.mouse.get_pos()
        draw_map.budget(trial, mouse_loc)
        
        if event.type == pg.QUIT:
            done = True
            
        elif event.type == pg.MOUSEMOTION:
            draw_map.budget(trial,mouse_loc)
            trial.static_data(mouse_loc,tick_second)
            
        elif event.type == pg.MOUSEBUTTONDOWN:
            draw_map.budget(trial,mouse_loc)
            if trial.check_end(): # not end
                trial.make_choice(mouse_loc)
                if trial.check == 1: # made valid choice
                    trial.budget_update()
                    trial.data(mouse_loc, tick_second)
                    draw_map.auto_snap(trial)
                else:
                    trial.static_data(mouse_loc,tick_second)
                    trial.click[-1] = 1
            else: # end
                print("The End") # need other end function
            
        elif event.type == pg.MOUSEBUTTONUP:
            draw_map.budget(trial,mouse_loc)
            trial.static_data(mouse_loc,tick_second)

        elif event.type == pg.KEYDOWN:
            if pg.key.get_pressed() and event.key == pg.K_z:
                trial.undo(mouse_loc, tick_second)
                print("budget undo" + str(trial.budget_dyn))
                
            if event.key == pg.K_ESCAPE:
                done = True   # very important, otherwise stuck in full screen
                pg.display.quit()
            if event.key == pg.K_RETURN:
                pg.event.set_blocked(pg.MOUSEMOTION)
                done = True

        elif event.type == pg.KEYUP:
            if event.key == pg.K_z:
                draw_map.budget(trial,mouse_loc)
                if len(trial.choice_dyn) >= 2:
                    draw_map.road(trial)

       
        pg.display.flip()  
        screen.fill(WHITE)
        draw_map = Draw(trial)

while done:
    for event in pg.event.get():
        screen.fill(WHITE)
        draw_map.game_end(trial)
        pg.display.flip() 
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                pg.display.quit()


# -------------------------------------------------------------------------
print("-----------------MAP INFORMATION --------------")
print("Starting City: " + str(trial.city_start))
print("city locations: " + str(trial.xy))
print("---------------- Break ----------------")
pg.quit()
