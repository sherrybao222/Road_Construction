import pygame as pg
import pandas as pd
import random
import math
from scipy.spatial import distance_matrix


# generate map and its corresponding parameters about people's choice
# -------------------------------------------------------------------------
class Map:
    def __init__(self): 
        
        self.uniform_map()
        self.data_init()
        
    def uniform_map(self):
        # map parameters
        self.N = 11     # total city number, including start
        self.radius = 7     # radius of city
        self.total = 700    # total budget
        self.budget_remain = 700    # remaining budget
        self.trial_n = 0

        # self.x = random.sample(range(500, 1400), self.N)    # x axis of all cities
        # self.y = random.sample(range(500, 1400), self.N)    # y axis of all cities
        map_file = pd.read_csv("trial.csv")
        map_row = map_file.loc[self.trial_n]
        print(map_row)
        self.xy = map_row[['City_xy']].tolist()
        # self.xy = self.map_col
        # [[self.x[i], self.y[i]] for i in range(0, len(self.x))]   # combine x and y
   
        self.city_start = map_row[['City_start']]
        print(self.city_start)
        # self.xy[0]    # start city
        self.distance = map_row[['Distance_m']]
        # distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix
    
        self.n_city = 0 # number of cities connected
        self.check = 0 # indicator showing if people made a valid choice
        
    # def map_load(self):
    #     map_file = pd.read_csv("trial.csv")
    #     map_row = map_file.loc[self.trial_n]
    #     self.map_col = map_row[['City_xy']]

    #     self.trial_num = 10
    #     self.map_file = pd.read_csv("trial.csv") # access the 4th_col of city_xy from "trial.csv", usecols=[3])
    #     for i in range(0, self.trial_num):
    #         self.map_row = self.map_file.loc[self.trial_num]
    #         self.map_col = self.map_row[['City_xy']]
    #         # print(map_col)

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
        
    def check_end(self): # check if trial end
        distance_copy = self.distance[self.choice_dyn[-1]] # copy distance list for current city
        if any(i < self.budget_dyn[-1] and i != 0 for i in distance_copy):
            return True # not end
        else:
            self.trial_n += 1
            return False # end
        
# visualize the game
# -------------------------------------------------------------------------
class Draw: 
    def __init__(self, mmap):
        self.instruction_submit()
        self.cities(mmap) # draw city dots
        if len(mmap.choice_dyn) >= 2: # if people have made choice, need to redraw the chosen path every time
            self.road(mmap)
        self.text_write("Score: " + str(mmap.n_city), 100, BLACK, 1600, 200) # show number of connected cities
         
    def road(self,mmap): # if people have made choice, need to redraw the chosen path every time
        pg.draw.lines(screen, BLACK, False, mmap.choice_locdyn, 5)

    def cities(self,mmap): # draw city dots       
        for city in mmap.xy[1:]: # exclude start city
            self.city = pg.draw.circle(screen, BLACK, city, 10)     
        self.start = pg.draw.circle(screen, RED, mmap.city_start, 10)
        
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

    def instruction_submit(self):
        self.text_write("Press Return to SUBMIT", 60, BLACK, 100, 200)

    def game_end(self, mmap): 
        #pg.draw.rect(screen, WHITE, (600, 600, 600, 200), 0)
        self.text_write('Your score is ' + str(mmap.n_city), 100, BLACK, 700, 750)
        pg.display.update()

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
next_trial = False

# display setup
screen = pg.display.set_mode((2000, 1500), flags=pg.FULLSCREEN)  # display surface
WHITE = (255, 255, 255)
RED = (255, 102, 102)
GREEN = (0, 204, 102)
BLACK = (0, 0, 0)
screen.fill(WHITE)
#clock = pg.time.Clock()
#FPS = 30  # tracking time check how to use
#clock.tick(FPS)
# -------------------------------------------------------------------------
trial = Map()
draw_map = Draw(trial)

# -------------------------------------------------------------------------
# loop for displaying until quit
while not done:
    for event in pg.event.get():
        tick_second = round((pg.time.get_ticks()/1000), 2)
        mouse_loc = pg.mouse.get_pos()
        
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
            if event.key == pg.K_ESCAPE:
                done = True   # very important, otherwise stuck in full screen
                pg.display.quit()
            if event.key == pg.K_RETURN:
                pg.event.set_blocked(pg.MOUSEMOTION)
                done = True
                # next_trial = True
                # draw_map.budget(map10.map_col, mouse_loc)

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

# while next_trial:
#     mouse_loc = pg.mouse.get_pos()
#     # draw_map.budget(trial.map_col, mouse_loc)
#     # done = False
#     # next_trial = False
#
#     for event in pg.event.get():
#         if event.type == pg.KEYDOWN:
#             if event.key == pg.K_ESCAPE:
#                 done = True  # very important, otherwise stuck in full screen
#                 pg.display.quit()
#             if event.key == pg.K_RETURN:
#                 pg.event.set_blocked(pg.MOUSEMOTION)
#                 done = True
#                 next_trial = True



# -------------------------------------------------------------------------
print("-----------------MAP INFORMATION --------------")
print("Starting City: " + str(trial.city_start))
print("city locations: " + str(trial.xy))
print("---------------- Break ----------------")
pg.quit()
