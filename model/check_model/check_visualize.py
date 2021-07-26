import pygame as pg
import pygame.gfxdraw
import math
from scipy.spatial import distance_matrix
import numpy as np
from best_first_search import make_move, new_node

# generate map and its corresponding parameters about people's choice
# ============================================================================
class Map:
    def __init__(self, map_content, trl_id, map_id): 
        
#        self.circle_map()
        self.load_map(map_content, map_id)
        self.data_init(trl_id, map_id)
       
#   different maps
# ----------------------------------------------------------------------------        
    def circle_map(self):
        # map parameters
        self.N = 30     # total city number, including start
        self.radius = 5     # radius of city
        self.total = 300    # total budget
        self.budget_remain = 300    # remaining budget
        
        self.R = 200*200 #circle radius' sqaure
        self.r = np.random.uniform(0, self.R, self.N).tolist() 
        self.phi = np.random.uniform(0,2 * math.pi, self.N).tolist()  
        self.x = np.sqrt(self.r) * np.cos(self.phi) + 1000
        self.x = self.x.astype(int).tolist() 
        self.y = np.sqrt(self.r) * np.sin(self.phi) + 800
        self.y = self.y.astype(int).tolist() 
        self.xy = [(self.x[i], self.y[i]) for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix
        self.dist_city = self.distance.copy()
                
        self.dict_city = dict(zip(list(range(0,self.N)), self.xy)) 
        self.dict_city_remain = self.dict_city.copy()

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
        self.y = [x + int(HEIGHT/2) for x in self.loadmap['y']]
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = self.loadmap['distance']      
        self.dist_city = self.distance.copy()
                
        self.dict_city = dict(zip(list(range(0,self.N)), self.xy)) 
        self.dict_city_remain = self.dict_city.copy()
          
# -----------------------------------------------------------------------------
    def make_choice(self):   
        
        self.choice = make_move(self.now,self.dist_city)
        self.city = self.xy[self.choice.name]
        
        self.choice_dyn.append(self.choice.name)
        self.choice_locdyn.append(self.city)
        self.choice_his.append(self.choice.name)
        self.choice_loc.append(self.city)
                
        self.budget_dyn.append(self.choice.budget)
        self.budget_his.append(self.choice.budget)
                                
        self.n_city.append(self.choice.n_c)
        print(self.choice.n_c)  
#        self.now = self.choice  
        
        self.new_start = new_node(self.choice.name, None, self.now.city, self.dist_city, self.choice.budget, self.now.n_c, [1,1,1])
        self.now = self.new_start

    def check_end(self): # check if trial end
        if self.now.determined:
            return False # end
        else:
            return True # not end  

# -----------------------------------------------------------------------------           
    def data_init(self, trl_id, map_id):
        self.now = new_node(0, None, self.dict_city_remain, self.dist_city, self.budget_remain, -1, [1,1,1])
        
        self.trl = [trl_id]
        self.mapid = [map_id]
        
        self.choice_dyn = [0]
        self.choice_locdyn = [self.city_start]
        self.choice_his = [0]   # choice history, index
        self.choice_loc = [self.city_start] # choice location history
                
        self.budget_dyn = [self.total]
        self.budget_his = [self.total] # budget history

        self.n_city = [0] # number of cities connected

        self.check_end_ind = 0

# score bar
# ============================================================================
class ScoreBar:
    def __init__(self,mmap):
        # only call once when initiated for this part
        # score bar parameters
        self.width = 100
        self.height = 480
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
        self.x = self.width / 2 + 1300 # larger the number, further to right

        for i in range(self.box):
            y =  i * self.box_height + self.uni_height
            loc = self.x, y
            self.center_list.append(loc)

    def incentive(self):
        self.score = list(range(0,self.box))
        self.incentive_score = []
        for i in self.score:
            i = (i ** 2)*0.01
            self.incentive_score.append(i)

    def indicator(self, mmap): # call this function to updates arrow location
        self.indicator_loc = self.center_list[mmap.n_city[-1]]
        self.indicator_loc_best = self.center_list[max(mmap.n_city)]
                                                                
# visualize the game
# ============================================================================
class Draw: 
    def __init__(self, mmap,screen,scorebar):
        self.budget(mmap, pg.mouse.get_pos(),screen)

        self.instruction_submit(screen)
        self.cities(mmap,screen) # draw city dots
        self.number(scorebar, screen)
        self.arrow(scorebar, screen)
        self.title(scorebar,screen)
        
        if len(mmap.choice_dyn) >= 2: # if people have made choice, need to redraw the chosen path every time
            self.road(mmap,screen)
        
#        self.text_write("Score: " + str(mmap.n_city[-1]), 100, BLACK, 1600, 200, screen) # show number of connected cities
        
        if mmap.check_end_ind:
             self.check_end(screen)
# -----------------------------------------------------------------------------                        
    def road(self,mmap,screen): # if people have made choice, need to redraw the chosen path every time
        for i in range(0,len(mmap.choice_locdyn)-1):
            draw_line(screen,mmap.choice_locdyn[i], mmap.choice_locdyn[i+1], BLACK)
            i = i + 1
# -----------------------------------------------------------------------------                        
    def cities(self,mmap,screen): # draw city dots       
        for city in mmap.xy[1:]: # exclude start city
            self.city = draw_circle(screen,city,mmap.radius,BLACK)
        self.start = draw_circle(screen,mmap.city_start,mmap.radius,RED)
                
# -----------------------------------------------------------------------------           
    def budget(self, mmap, mouse,screen):  
        # current mouse position
        cx, cy = mouse[0] - mmap.choice_locdyn[-1][0], mouse[1] - mmap.choice_locdyn[-1][1]
        # give budget line follow mouse in the correct direction
        radians = math.atan2(cy, cx)
        budget_pos = [int(mmap.choice_locdyn[-1][0] + mmap.budget_dyn[-1] * math.cos(radians)),
                      int(mmap.choice_locdyn[-1][1] + mmap.budget_dyn[-1] * math.sin(radians))]
        self.budget_line = pg.draw.line(screen, GREEN, mmap.choice_locdyn[-1], budget_pos, 4)
#        self.budget_line = draw_line(screen,mmap.choice_locdyn[-1], budget_pos, GREEN)

# -----------------------------------------------------------------------------                   
    def auto_snap(self, mmap,screen):
        pg.draw.line(screen, BLACK, mmap.choice_locdyn[-2], mmap.choice_locdyn[-1], 3)

# -----------------------------------------------------------------------------           
    def check_end(self,screen):
        text_write("Out of budget", 60, RED, 100, 300,screen)

    def instruction_submit(self,screen):
        text_write("Press RETURN to SUBMIT", 60, BLACK, 100, 200, screen)
# ---------------------------------------------------------------------------------
    def number(self, scorebar, screen):
        left = scorebar.center_list[0][0] - 25
        c_list = [(102,204,102),(116,195,102),(130,185,102),(144,176,102),
                  (158,167,102),(172,158,102),(185,148,102),(199,139,102),
                  (213,130,102),(227,121,102),(241,111,102),(255,102,102)] # color list
        for i in range(scorebar.box):
            loc = scorebar.center_list[i]
            text = scorebar.incentive_score[i]
            pg.draw.rect(screen, c_list[i],
                         (left, loc[1] + scorebar.top - scorebar.uni_height,
                          scorebar.width, scorebar.box_height), 0) # if width == 0, (default) fill the rectangle
            pg.draw.rect(screen, BLACK, (left, loc[1]+scorebar.top-scorebar.uni_height, 
                                         scorebar.width, scorebar.box_height), 2)  # width for line thickness
            text_write(str(text), int(scorebar.box_height - 10), BLACK, loc[0], loc[1]+scorebar.top , screen) # larger number, further to right
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

    def title(self, scorebar,screen):
        x = scorebar.center_list[0][0]-20
        y = scorebar.center_list[0][1]+scorebar.top-60
        text_write("Bonus in dollars", 50, BLACK, x, y, screen)

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
def pygame_trial(all_done, trl_done, map_content, trl_id, screen, map_id):
    
    trial = Map(map_content, trl_id, map_id)
    scorebar = ScoreBar(trial)
    
    while not trl_done:
        
        screen.fill(GREY)
        draw_map = Draw(trial,screen,scorebar)
        pg.display.flip()  

        for event in pg.event.get():
            
            if event.type == pg.QUIT:
                all_done = True
               
            elif event.type == pg.MOUSEBUTTONDOWN:
                if trial.check_end(): # not end
                    trial.make_choice()
                    draw_map.auto_snap(trial,screen)  
                    scorebar.indicator(trial)
                else: # end
                    print("The End") # need other end function
                
            elif event.type == pg.MOUSEBUTTONUP:
                if not trial.check_end(): 
                    trial.check_end_ind = 1
                    
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    all_done = True   # very important, otherwise stuck in full screen
                    pg.display.quit()
                    pg.quit()
                if event.key == pg.K_RETURN and trial.n_city[-1] != 0:
#                    pg.event.set_blocked(pg.MOUSEMOTION)
                    trl_done = True
                    break
                
    return all_done,trial,trl_done

# multiple trials
# =============================================================================
def road_basic(screen,map_content,n_trials):
     # conditions
    all_done = False
    trl_done = False
    
    trials = []
    
    # -------------------------------------------------------------------------
    while not all_done:
        for trl_id in range(0, n_trials):
            all_done,trial,trl_done = pygame_trial(all_done, trl_done, map_content, 
                                                   trl_id + 1, screen, trl_id + 1)
            trl_done = False 
            trials.append(trial)
        all_done = True
        
# saving
#    sio.savemat('test_saving_basic.mat', {'trials':trials}) 
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
    screen = pg.display.set_mode((WIDTH, HEIGHT))#, flags=pg.FULLSCREEN)  # pg.FULLSCREEN pg.RESIZABLE
        
    # Fill background 
    screen.fill(GREY)

    # load maps
    import json
    #with open('/Users/fqx/Spring 2020/Ma Lab/GitHub/Road_Construction/map/basic_map_24','r') as file:
    with open('/Users/sherrybao/Downloads/research/road_construction/rc_all_data/map/active_map/basic_map_48_all4','r') as file: 
        map_content = json.load(file)[0] 
    
    
    n_trials = 48
 
    trials = road_basic(screen,map_content,n_trials)
    
    pg.display.quit()
    
    pg.quit()

