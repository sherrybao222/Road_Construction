import pygame as pg
import math
from scipy.spatial import distance_matrix
import numpy as np

# generate map and its corresponding parameters about people's choice
# ============================================================================
class Map:
    def __init__(self, trl_id): 
        
        self.circle_map()
        self.data_init(trl_id)
       
#   different maps
# ----------------------------------------------------------------------------        
    def circle_map(self):
        # map parameters
        self.N = 20     # total city number, including start
        self.radius = 5     # radius of city
        
        self.total = 400    # total budget
        self.budget_remain = 400    # remaining budget
        self.total_a = 400 # agent
        self.budget_remain_a = 400

        self.R = 300*300 #circle radius' sqaure
        self.r = np.random.uniform(0, self.R, self.N) 
        self.phi = np.random.uniform(0,2 * math.pi, self.N) 
        self.x = np.sqrt(self.r) * np.cos(self.phi) + 1000
        self.x = self.x.astype(int)
        self.y = np.sqrt(self.r) * np.sin(self.phi) + 800
        self.y = self.y.astype(int)
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city of people
#        self.city_start_a = self.xy[1]    # start city of agent
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix
        self.matrix_copy = self.distance.copy()
        
        dist_list = self.matrix_copy[0] # choose the related column/row
        dist = np.amin(dist_list[dist_list != 0]) # the smallest non-zero distance          
        self.city_start_a_ind = np.where(dist_list == dist)[0][0] # find the chosen city index
        self.city_start_a = self.xy[self.city_start_a_ind]

        self.matrix_copy[:, 0] = 0 # cannot choose one city twice
        self.matrix_copy[0, :] = 0      
        
# human          
# -----------------------------------------------------------------------------        
    def make_choice(self, mouse):
        for i in range(1, self.N): # do not evaluate the starting point
            x2, y2 = mouse # mouse location
            self.mouse_distance = math.hypot(self.x[i] - x2, self.y[i] - y2)
            if (self.mouse_distance <= self.radius) and (i not in self.choice_dyn 
                                                           and i not in self.choice_dyn_a): # cannot choose what has been chosen
                self.index = i # index of chosen city
                self.city = self.xy[i] # location of chosen city
                self.check = 1 # indicator showing people made a valid choice
                
                self.matrix_copy[:, self.index] = 0 # cannot choose one city twice
                self.matrix_copy[self.index, :] = 0
        
    def budget_update(self):
        dist = self.distance[self.index][self.choice_dyn[-1]] # get distance from current choice to previous choice
        self.budget_remain = self.budget_dyn[-1] - dist # budget update
       
    def check_end(self): # check if trial end
        distance_copy = self.distance[self.choice_dyn[-1]].copy() # copy distance list for current city
        for x in self.choice_dyn:
            distance_copy[x] = 0
        for x in self.choice_dyn_a:
            distance_copy[x] = 0
        if any(i < self.budget_dyn[-1] and i != 0 for i in distance_copy):
            return True # not end
        else:
            return False # not end
        
# agent
# -----------------------------------------------------------------------------
    def make_choice_a(self):        
        dist_list = self.matrix_copy[self.choice_dyn_a[-1]] # choose the related column/row
        dist = np.amin(dist_list[dist_list != 0]) # the smallest non-zero distance

            
        self.index_a = np.where(dist_list == dist)[0][0] # find the chosen city index
        self.city_a = self.xy[self.index_a]
        self.matrix_copy[:, self.choice_dyn_a[-1]] = 0 # cannot choose one city twice
        self.matrix_copy[self.choice_dyn_a[-1], :] = 0

        dist = self.distance[self.index_a][self.choice_dyn_a[-1]] # get distance from current choice to previous choice
        self.budget_remain_a = self.budget_dyn_a[-1] - dist  # budget update
        
        self.choice_dyn_a.append(self.index_a)
        self.choice_locdyn_a.append(self.city_a)
        self.choice_his_a.append(self.index_a)
        self.choice_loc_a.append(self.city_a)
                
        self.budget_dyn_a.append(self.budget_remain_a)
        self.budget_his_a.append(self.budget_remain_a)
                                
        self.n_city_a.append(self.n_city_a[-1] + 1)
                
    def check_end_a(self): # check if trial end
        dist_list = self.distance[self.choice_dyn_a[-1]] # choose the related column/row
        dist = np.amin(dist_list[dist_list != 0]) # the smallest non-zero distance
        
        if (self.budget_remain_a - dist < 0):
            return False # end
        else:
            return True # not end  
        
# data saving structure
# -----------------------------------------------------------------------------           
    def data_init(self, trl_id):
        self.trl = [trl_id]
        self.cond = [2] # condition
        self.time = [round((pg.time.get_ticks()/1000), 2)] # mouse click time 
        self.pos = [pg.mouse.get_pos()]
        self.click = [0] # mouse click indicator
        
        self.choice_dyn = [0]
        self.choice_locdyn = [self.city_start]
        self.choice_his = [0]   # choice history, index
        self.choice_loc = [self.city_start] # choice location history
        
        self.choice_dyn_a = [self.city_start_a_ind] # computer
        self.choice_locdyn_a = [self.city_start_a]
        self.choice_his_a = [self.city_start_a_ind]   # choice history, index
        self.choice_loc_a = [self.city_start_a] # choice location history

                
        self.budget_dyn = [self.total]
        self.budget_his = [self.total] # budget history
        
        self.budget_dyn_a = [self.total_a]
        self.budget_his_a = [self.total_a] # budget history


        self.n_city = [0] # number of cities connected
        self.n_city_a = [0]
        self.check = 0 # indicator showing if people made a valid choice

        self.check_end_ind = 0
        self.check_end_a_ind = 0

    def data(self, mouse, time, trl_id): 
        self.trl.append(trl_id)
        self.cond.append(2)
        self.time.append(time)
        self.pos.append(mouse)
        self.click.append(1)
        
        self.choice_dyn.append(self.index)
        self.choice_locdyn.append(self.city)
        self.choice_his.append(self.index)
        self.choice_loc.append(self.city)
                
        self.budget_dyn.append(self.budget_remain)
        self.budget_his.append(self.budget_remain)
                                
        self.n_city.append(self.n_city[-1] + 1)
        self.check = 0 # change choice indicator after saving them

        del self.index, self.city   
        
    def static_data(self, mouse, time, trl_id): 
        self.trl.append(trl_id)
        self.cond.append(2)
        self.time.append(time)
        self.pos.append(mouse)
        self.click.append(0)
        
        self.choice_his.append(self.choice_dyn[-1])
        self.choice_loc.append(self.choice_locdyn[-1])  
        self.budget_his.append(self.budget_dyn[-1])
        
        self.n_city.append(self.n_city[-1])
        
# visualize the game
# ============================================================================
class Draw: 
    def __init__(self, mmap,screen):
        self.budget(mmap, pg.mouse.get_pos(),screen)
        self.budget_a(mmap, screen)

        self.instruction_submit(screen)
        self.cities(mmap,screen) # draw city dots
        if len(mmap.choice_dyn) >= 2: # if people have made choice, need to redraw the chosen path every time
            self.road(mmap,screen)
        if len(mmap.choice_dyn_a) >= 2: # if people have made choice, need to redraw the chosen path every time
            self.road_a(mmap,screen)
        
        self.text_write("Score: " + str(mmap.n_city[-1] + mmap.n_city_a[-1]), 100, BLACK, 1600, 200, screen) # show number of connected cities
        
        if mmap.check_end_a_ind:
             self.check_end_a(screen)
        if mmap.check_end_ind:
             self.check_end(screen)
# -----------------------------------------------------------------------------                        
    def cities(self,mmap,screen): # draw city dots       
        for city in mmap.xy[1:]: # exclude start city
            self.city = pg.draw.circle(screen, BLACK, city, mmap.radius)     
        self.start = pg.draw.circle(screen, RED, mmap.city_start, mmap.radius)
        self.start = pg.draw.circle(screen, PINK, mmap.city_start_a, mmap.radius)
# -----------------------------------------------------------------------------                        
    def road(self,mmap,screen): # if people have made choice, need to redraw the chosen path every time
        pg.draw.lines(screen, BLACK, False, mmap.choice_locdyn, 3)
        
    def road_a(self,mmap,screen): # if people have made choice, need to redraw the chosen path every time
        pg.draw.lines(screen, BROWN, False, mmap.choice_locdyn_a, 3)
        
# -----------------------------------------------------------------------------           
    def budget(self, mmap, mouse,screen):  
        # current mouse position
        cx, cy = mouse[0] - mmap.choice_locdyn[-1][0], mouse[1] - mmap.choice_locdyn[-1][1]
        # give budget line follow mouse in the correct direction
        radians = math.atan2(cy, cx)
        budget_pos = (int(mmap.choice_locdyn[-1][0] + mmap.budget_dyn[-1] * math.cos(radians)),
                      int(mmap.choice_locdyn[-1][1] + mmap.budget_dyn[-1] * math.sin(radians)))
        self.budget_line = pg.draw.line(screen, GREEN, mmap.choice_locdyn[-1], budget_pos, 3)

    def budget_a(self, mmap, screen):  
        # current sudo mouse position
        if len(mmap.choice_dyn_a) <= 1:
            cx, cy = 0, 1
        else:
            cx, cy = (mmap.choice_locdyn_a[-1][0] - mmap.choice_locdyn_a[-2][0], 
                     mmap.choice_locdyn_a[-1][1] - mmap.choice_locdyn_a[-2][1])                   
        # give budget line follow sudo mouse in the correct direction
        radians = math.atan2(cy, cx)
        budget_pos = (int(mmap.choice_locdyn_a[-1][0] + mmap.budget_dyn_a[-1] * math.cos(radians)),
                      int(mmap.choice_locdyn_a[-1][1] + mmap.budget_dyn_a[-1] * math.sin(radians)))
        self.budget_line = pg.draw.line(screen, BLUE, mmap.choice_locdyn_a[-1], budget_pos, 3)

# -----------------------------------------------------------------------------           
    def auto_snap(self, mmap,screen):
        pg.draw.line(screen, BLACK, mmap.choice_locdyn[-2], mmap.choice_locdyn[-1], 3)
        
    def auto_snap_a(self, mmap,screen):
        pg.draw.line(screen, BROWN, mmap.choice_locdyn_a[-2], mmap.choice_locdyn_a[-1], 3)

# -----------------------------------------------------------------------------           
    def check_end(self,screen):
        self.text_write("You are out of budget", 60, RED, 100, 400,screen)

    def check_end_a(self,screen):
        self.text_write("Your partner is out of budget", 60, RED, 100, 300,screen)

    def instruction_submit(self,screen):
        self.text_write("You are controlling the green budget", 60, BLACK, 100, 100,screen)
        self.text_write("Press Return to SUBMIT", 60, BLACK, 100, 200,screen)

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
def pygame_trial(all_done, trl_done, trl_id, screen):
    
    trial = Map(trl_id)
    pg.display.flip()
    screen.fill(WHITE)
    draw_map = Draw(trial,screen)
    your_turn = True
    marker = 0    
    
    while not trl_done:
        
        if your_turn == False:
           
            if trial.check_end_a():               
                pg.time.delay(1200)               
                trial.make_choice_a()
                draw_map.auto_snap_a(trial,screen) 
                if not trial.check_end_a():  
                    trial.check_end_a_ind = 1
            else:
                trial.check_end_a_ind = 1
                
            if  trial.check_end(): 
                your_turn = True
            else:
                trial.check_end_ind = 1
                your_turn = False
                
          
        screen.fill(WHITE)
        draw_map = Draw(trial,screen)
        pg.display.flip()  

            
        if marker == 1:
            your_turn = False
            marker = 0

                
        for event in pg.event.get():
            tick_second = round((pg.time.get_ticks()/1000), 2)
            mouse_loc = pg.mouse.get_pos()
                                                
            if event.type == pg.QUIT:
                all_done = True
    
            elif event.type == pg.MOUSEMOTION:
                trial.static_data(mouse_loc,tick_second,trl_id)
           
            elif event.type == pg.MOUSEBUTTONDOWN:
                trial.click[-1] = 1
                if your_turn == True:
                    if trial.check_end(): # not end
                        trial.make_choice(mouse_loc)
                        if trial.check == 1: # made valid choice
                            trial.budget_update()
                            trial.data(mouse_loc, tick_second, trl_id)
                            draw_map.auto_snap(trial,screen) 
                            marker = 1
                            
                        else:
                            trial.static_data(mouse_loc, tick_second, trl_id)
                    else: # end
                        trial.check_end_ind = 1
                    
                
            elif event.type == pg.MOUSEBUTTONUP:
                trial.static_data(mouse_loc, tick_second, trl_id)
                if  not trial.check_end(): 
                    trial.check_end_ind = 1
                        
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    all_done = True   # very important, otherwise stuck in full screen
                    pg.quit()
                if event.key == pg.K_RETURN and trial.n_city[-1] != 0:
                    trl_done = True
                    break

        
        screen.fill(WHITE)
        draw_map = Draw(trial,screen)
        pg.display.flip()  
    
    return all_done,trial,trl_done

# multiple trials
# =============================================================================
def road_basic(screen,n_trials):
     # conditions
    all_done = False
    trl_done = False
    
    trials = []
    
    # -------------------------------------------------------------------------
    while not all_done:
        for trl_id in range(0, n_trials):
            all_done,trial,trl_done = pygame_trial(all_done, trl_done,
                                                   trl_id + 1, screen)
            trl_done = False 
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
BROWN = (102, 51, 0)
PINK = (255, 153, 204)
BLUE = (51, 153, 255)

if __name__ == "__main__":
    pg.init()
    pg.font.init()
      
    # display setup
    screen = pg.display.set_mode((2000, 1600), flags=pg.FULLSCREEN)  # pg.FULLSCREEN pg.RESIZABLE
        
    # Fill background 
    screen.fill(WHITE)
    
    
    n_trials = 5
 
    trials = road_basic(screen,n_trials)
    
    pg.quit()
