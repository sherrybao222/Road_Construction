import pygame as pg
import random
import math

# -------------------------------------------------------------------------
# A list of Class objects: City, Map, Budget, DrawBoard

# Generate individual city coordinates and size
class City:
    def __init__(self): # 注意多了一个argument，代入city
        self.N = 11 #total city number with starting location
        self.radius = 7
        self.x = random.sample(range(51, 649), self.N)  # set up in a way so don't overlap with budget information
        self.y = random.sample(range(51, 649), self.N)
        self.map = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]
        # 这个directory 名字改了，后面的都要找这个map
      	self.city_start = self.map[0] # the starting city from the list
        self.distance = sp.spatial.distance_matrix(self.map, self.map, p=2, threshold=10000)

# -------------------------------------------------------------------------
# anything relevant to budget, total and history of past budget
class Budget:
    def __init__(self):
        self.total = 700  # can change this total budget

    # 这个主要检测用户有没有点在我们给的城市上面
    @staticmethod  # >A< input changed, need mouse and current map type, return the selected city
    def collision(mouse_loc, city):
        for i in range(1, city.N):
            x2, y2 = mouse_loc  # pg.mouse.get_pos()
            distance = math.hypot(city.x[i] - x2, city.y[i] - y2)
            if distance <= city.radius:
                return i, city.map[i]


# calculate current budget using the distance between selected cities
    @staticmethod
    def budget_update(data, city):
        budget = data.budget_his[-1]  # the latest remain budget from data saving
        city_b = data.choice[-1][0]  # find the paired city from user choice lists
        city_a = data.choice[-2][0]
        i = city.map.index(city_b) # convert to index of those city for this map
        j = city.map.index(city_a)
        distance = city.distance[i, j]  # using index to find distance from matrix
        return round((budget - distance), 2)

# 这个主要是解决budget rotation的问题。因为pygame划线需要两点一线，但那个终点是跟着鼠标移动的
# 所以这个像reverse engineering，反向求出终点然后画budget line
# 最后给出来的是budget line 的end point
    @staticmethod   # given loc_a & distance for loc_b to draw line
    def budget_pos(data):
        city_x = data.choice[-1][0]
        city_y = data.choice[-1][1]     # xy = current city location xy
        cx, cy = pg.mouse.get_pos()[0] - city_x, pg.mouse.get_pos()[1] - city_y
            # cx, cy = pg.mouse.get_pos()[0] - click[-1][0], pg.mouse.get_pos()[1] - click[-1][1]
        radians = math.atan2(cy, cx)    #budget direction given mouse position
        d = data.budget_his[-1]     # d = budget remain
        budget_pos = int(city_x + d * math.cos(radians)), int(city_y + d * math.sin(radians))
            # the end point of budget line given budget left & mouse position
            # print("budget_dis: " + str(Map.distance(click[-1], budget_pos)), "budget: " + str(budget.bud_history[-1]))
            # print("-------------------------------")
        return budget_pos

# 这个相当于undo之后把之前的budget还给你，顺便那之前的选择给消除了
# 消除的是顶上那两个memory list，所以过程和data是存在另一个地方的
    @staticmethod
    def budget_undo():
        undo = round(distance_his[-1] + budget.bud_history[-1], 2)
        budget.bud_history.append(undo)
        distance_his.pop(-1)
        click.pop(-1)
        pg.draw.line(screen, WHITE, click[-2], click[-1], 3)

# -------------------------------------------------------------------------
# 这个主要存所以的input，包括mouse location，时间之类的
class Data:
    def __init__(self):
        self.choice = [0]
        self.click_his = [(0, 0)]
        self.click_time = []  # record the click time since game started
        self.movement = []  # get.rel
        self.budget_his = []

    @staticmethod  # find a way to code to adding to new instance, check arguement
    def saving(trial, budget):
        trial.choice.append(budget.collision(mouse_loc, city))
        trial.click_his.append(pg.mouse.get_pos())
        trial.movement.append(pg.mouse.get_rel())
        tick_second = round((pg.time.get_ticks() / 1000), 2)
        trial.click_time.append(tick_second)

# 这个是整体刷新，就是当被试点了city之后，所有相关的distance/budget的后台数据
    @staticmethod
    def game_update():  # update with input for new draw information
        click.append(pg.mouse.get_pos())
        distance_his.append(Map.distance(click[-2], click[-1]))
        map_1.dis_history.append(Map.distance(click[-2], click[-1]))
        budget.bud_history.append(Budget.budget_update(budget.bud_history[-1], map_1.dis_history[-1]))


# -------------------------------------------------------------------------
# 这个是主要的前端设计，包括各种visualization 和功能
class Draw:

    # 一旦鼠标接近城市，这个就会自动连线，并覆盖等量的budget line
    @staticmethod
    def auto_snap(mouse_loc):
        budget_allow = budget.bud_history[-1] >= Map.distance(click[-1], pg.mouse.get_pos())
        if budget_allow:
            if Draw.collision(pg.mouse.get_pos()):
                # this is a bug, somehow need to change Budget_allow calculation
                pg.draw.line(screen, BLACK, click[-1], mouse_loc, 3)
                # print("||||||||||||||||||||||||||||||||||||||||")
                # print("bud_his: " + str(budget.bud_history[-1]), "dis: " + str(Map.distance(click[-1], pg.mouse.get_pos())))
                # print("Allow?: " + str(budget.bud_history[-1] > Map.distance(click[-1], pg.mouse.get_pos())))

    # 这个主要确定budget line上面的那个滑点停在最终点
    # 而不是跟着鼠标乱跑
    @staticmethod
    def mouse_limit(mouse_loc):
        mouse_dis = Map.distance(click[-1], mouse_loc)
        if mouse_dis < budget.bud_history[-1]:
            pg.draw.circle(screen, GREEN, pg.mouse.get_pos(), 5)
        else:
            pg.draw.circle(screen, GREEN, Budget.budget_pos(click[-1][0], click[-1][1], budget.bud_history[-1]), 5)

    # 这个是road construction，连续画线，从一个list里面顺着连那种
    @staticmethod
    def road_visual():
        pg.draw.lines(screen, BLACK, False, click, 1)
        pg.draw.line(screen, WHITE, click[0], city_start.xy, 3)
        # to hide the first line draw from origin
        pg.draw.circle(screen, RED, city_start.xy, 10)

    # undo box的visualization
    @staticmethod
    def road_undo():
        Draw.text_write("Undo", 50, BLACK, 900, 600)
        pg.draw.rect(screen, GREEN, (900, 600, 100, 50), 3)
        # those variable should be set at the top, so it's obvious

    # reset box的visualization
    @staticmethod
    def road_reset():
        Draw.text_write("Reset", 50, BLACK, 900, 550)
        pg.draw.rect(screen, GREEN, (900, 550, 100, 50), 3)

    # 写字的general function
    @staticmethod
    def text_write(text, size, color, x, y):  # function that can display any text
        font_object = pg.font.SysFont(pg.font.get_default_font(), size)
        text_surface = font_object.render(text, True, color)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = x, y
        screen.blit(text_surface, text_rectangle.center)

    # 主要的draw initiation，把所有部分都拼凑到一起
    # 这一片的规划也有点乱
    def __init__(self):
        self.undo_box = pg.draw.rect(screen, GREEN, (900, 600, 100, 50), 3)
        self.rest_box = pg.draw.rect(screen, GREEN, (900, 550, 100, 50), 3)
        Draw.road_visual()
        Draw.mouse_limit(pg.mouse.get_pos())
        Draw.road_undo()
        Draw.road_reset()
        city_visit = len(click) - 2
        # find a better structure to place this background information
        Draw.text_write(str(city_visit), 100, BLACK, 900, 100)
        for cities in map_1.all_city_xy:
            self.city_rect = (pg.draw.circle(screen, BLACK, cities, 7))
        pg.draw.line(screen, GREEN, click[-1], Budget.budget_pos(click[-1][0], click[-1][1], budget.bud_history[-1]), 3)
        if budget.bud_history[-1] > Map.distance(click[-1], pg.mouse.get_pos()):
            Draw.auto_snap(pg.mouse.get_pos())
            # this is still a bug
        pg.display.flip()
        screen.fill(WHITE)
        clock.tick(FPS)

    # 这个就是reset之后，把那个memory list刷新重启
    # 同时在data 里面标注0来表示用户重启
    @staticmethod
    def refresh():  # append put anchor to indicate new start
        click.clear()
        click.append((0, 0))
        click.append(city_start.xy)
        pg.mouse.set_pos(city_start.xy)
        budget.bud_history.append(budget.total)
        data1.movement.append((0, 0))
        data1.click_time.append(0)
        draw_map.__init__()
        pg.display.flip()

#------------------------------------------------------------------------------
#   大问题
# 结构我觉得有点乱，不是特别简明。而且我想用distance matrix赶紧会更方便，但是不确定该怎么写
# budget，city range，和screen的设定很微妙，因为会有情况看不到完整的budget
# data saving/data structure 我也不是很懂，目前我只会把所有东西都堆在一起
# budget line 永远向右指那个我不知道是bug还是怎么回事

#   setting up window, basic features 
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

# current memory for running the game, not saving (updating on current location)
click = [(0, 0)]
distance_his = [0]

# -------------------------------------------------------------------------
# 这一部分贼乱，我可能不是特别懂class和object的关系，所以我想重新写一个直接用class的版本
# 那个版本和distance matrix都在同一个草稿里，但是还没写完，也不知道写对没有
# those should be good order, otherwise will generate twice and don't know which to use
city_start = City()
map_1 = Map()
budget = Budget()
click.append(city_start.xy)
# a lot of places use this temporary list for update, but it's different from saving

# -------------------------------------------------------------------------
# class objects
draw_map = Draw()
mouse = pg.mouse.get_pos()

# 这个就是game loop，各种condition
# 主要分成mousemotion = 用户探索
# 和mousebuttondown = 用户决策 两部分

data1 = Data()

# -------------------------------------------------------------------------
# loop for displaying until quit
while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
        elif event.type == pg.MOUSEMOTION:
            draw_map.__init__()
        elif event.type == pg.MOUSEBUTTONDOWN:
            if draw_map.collision(pg.mouse.get_pos()) and event.button == 1:
                if budget.bud_history[-1] >= Map.distance(click[-2], click[-1]):
                    data1.game_update()
                    data1.saving()
                    draw_map.__init__()
                else:
                    # budget.bud_history[-1] <= Map.distance(click[-1], pg.mouse.get_pos()):
                    draw_map.refresh()
            if pg.Rect.collidepoint(draw_map.undo_box, pg.mouse.get_pos()[0], pg.mouse.get_pos()[1]) and event.button == 1:
                budget.budget_undo()
                # click.pop(-1)
                # # budget.bud_history.append(budget.bud_history[-2]) # bug to give back used budget
                # pg.draw.line(screen, WHITE, click[-2], click[-1], 3)
            if pg.Rect.collidepoint(draw_map.rest_box, pg.mouse.get_pos()[0], pg.mouse.get_pos()[1]) and event.button == 1:
                Draw.refresh()



# -------------------------------------------------------------------------
print("-----------------MAP INFORMATION --------------")
print("Starting City: " + str(city_start.xy))
print("city locations: " + str(map_1.all_city_xy))
print("---------------- INPUT DATA ----------------")
print("click history Data: " + str(data1.click_his))
print("click memory: " + str(click))
print("distance history Data: " + str(map_1.dis_history))
print("distance memory: " + str(distance_his))
print("budget history: " + str(budget.bud_history))
print("---------------- Break ----------------")
pg.quit()
