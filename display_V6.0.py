import pygame as pg
import random
import math
import scipy.spatial
from anytree import Node

# -------------------------------------------------------------------------
class Map:
    def __init__(self): 
        self.N = 11 # total city number
        self.radius = 7 # radius of city
        self.budget_remain = 700
        self.total = 700 # total budget
        
        self.x = random.sample(range(51, 649), self.N)  
        self.y = random.sample(range(51, 649), self.N)
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]
   
        self.city_start = self.xy[0] # the starting city from the list
        self.distance = scipy.spatial.distance_matrix(self.xy, self.xy, p=2, threshold=10000)
 
    
    
        self.choice = Node(0, budget = self.total) # start point
        self.click = []
        self.click_time = []    
        #self.movement = []  
        self.budget_his = [self.total]
        self.choice_his = [0]
        self.choice_loc = [self.city_start]
        self.n_city = 0
        self.check = 0
    
    def make_choice(self, mouse):
        for i in range(1, self.N): # do not evaluate the starting point
            x2, y2 = mouse  
            self.mouse_distance = math.hypot(self.x[i] - x2, self.y[i] - y2)
            if (self.mouse_distance <= self.radius) and (i not in self.choice_his):
                self.index = i
                self.city = self.xy[i]
                self.check = 1
        

    def budget_update(self):
        dist =  self.distance[self.index][self.choice_his[-1]] # the latest remain budget from data saving
        self.budget_remain = self.budget_remain - dist
       
        # using index to find distance from matrix
            
   
    def data(self, mouse):
        tick_second = round((pg.time.get_ticks()/1000), 2)
        self.click_time.append(tick_second)
        self.click.append(mouse)
        #mmap.movement.append(pg.mouse.get_rel()) 
        self.budget_his.append(self.budget_remain)
        self.choice_his.append(self.index)
        self.choice_loc.append(self.city)
        new = Node(self.index, parent = self.choice, budget = self.budget_remain, time = tick_second)
        
        self.n_city = self.n_city + 1
        self.choice = new
        
        self.check = 0 
        del self.index, self.city
        
    def check_end(self):
        distance_copy = self.distance[self.choice_his[-1]]
        distance_copy[self.choice_his[-1]] = 100000
        if all(i > self.budget_his[-1] and i != 0 for i in distance_copy):
            return False
        else:
            return True
        
    def undo(self):
        new = self.choice.parent
        self.choice = new
        budget = self.budget_his[-2]
        self.budget_his.append(budget)
        choice = self.choice_his[-2]
        self.choice_his.append(choice)
        

# -------------------------------------------------------------------------
class Draw:
    
    def __init__(self, mmap, mouse):
        self.cities(mmap)
        #Draw.undo_box()
        if len(mmap.choice_his) >= 2:
            self.road(mmap)
        self.text_write(str(mmap.n_city), 100, BLACK, 900, 100) 
        
        
        
    def road(self,mmap):
        pg.draw.lines(screen, BLACK, False, mmap.choice_loc, 1)

    
    def cities(self, mmap):
        
        for city in mmap.xy[1:]:
            self.city = pg.draw.circle(screen, BLACK, city, 7)
           
        self.start = pg.draw.circle(screen, RED, mmap.city_start, 7)
        
        
    # given loc_a & distance for loc_b to draw line
    def budget(self, mmap, mouse):  # d = budget remain bud_his[-1], xy = current city loc city[-1][0], [-1][1]
        cx, cy = mouse[0] - mmap.x[mmap.choice_his[-1]], mouse[1] - mmap.y[mmap.choice_his[-1]]
        # current mouse position
        radians = math.atan2(cy, cx)
        # give budget line follow mouse in the correct direction
        budget_pos = (int(mmap.x[mmap.choice_his[-1]] + mmap.budget_his[-1] * math.cos(radians)), 
                      int(mmap.y[mmap.choice_his[-1]] + mmap.budget_his[-1] * math.sin(radians)))

        self.budget_line = pg.draw.line(screen, GREEN, mmap.xy[mmap.choice_his[-1]], budget_pos, 3)
    
      
    def auto_snap(self, mmap):
        pg.draw.line(screen, BLACK, mmap.xy[mmap.choice_his[-2]], 
                     mmap.xy[mmap.choice_his[-1]], 3)
  
    #@staticmethod
    #def undo_box():
    #   Draw.text_write(self, "Undo", 50, BLACK, 900, 600)
    #    pg.draw.rect(screen, GREEN, (900, 600, 100, 50), 3)
        # those variable should be set at the top, so it's obvious

   
    def text_write(self, text, size, color, x, y):  # function that can display any text
        font_object = pg.font.SysFont(pg.font.get_default_font(), size)
        text_surface = font_object.render(text, True, color)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = x, y
        screen.blit(text_surface, text_rectangle.center)


#------------------------------------------------------------------------------
trial = Map()
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

mouse_loc = trial.city_start
draw_map = Draw(trial,mouse_loc)

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
            if trial.check_end():
                trial.make_choice(mouse_loc)
                if trial.check == 1:
                    trial.budget_update()
                    trial.data(mouse_loc)
                    draw_map.auto_snap(trial)
            if not(trial.check_end()):
                print("The End")
            #if pg.Rect.collidepoint(draw_map.undo_box, pg.mouse.get_pos()[0], pg.mouse.get_pos()[1]) and event.button == 1:
            #    budget.budget_undo()
        if event.type == pg.MOUSEBUTTONUP:
            draw_map.budget(trial,pg.mouse.get_pos())
            
        pg.display.flip()  
        screen.fill(WHITE)
        draw_map = Draw(trial,pg.mouse.get_pos())


# -------------------------------------------------------------------------
print("-----------------MAP INFORMATION --------------")
print("Starting City: " + str(trial.city_start))
print("city locations: " + str(trial.xy))
print("---------------- Break ----------------")
pg.quit()
