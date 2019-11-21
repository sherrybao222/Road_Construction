import pygame as pg
import random
import math

pg.init()
pg.font.init()
# conditions
done = False
# display setup
screen = pg.display.set_mode((2000, 1500), flags=pg.FULLSCREEN)   # display surface
clock = pg.time.Clock()
FPS = 30  # tracking time check how to use
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
screen.fill(WHITE)

# current memory for running the game, not saving
click = [(0, 0)]
distance_his = [0]
# A list of Class objects: City, Map, Budget, DrawBoard


class City:

    def __init__(self):
        self.x = random.randint(500, 1400)  # set up in a way so don't overlap with budget information
        self.y = random.randint(500, 1400)
        self.xy = self.x, self.y


class Map:
    def __init__(self):
        self.cities = [City() for i in range(10)]
        self.all_city_xy = [city.xy for city in self.cities]
        self.dis_history = [0]

    @staticmethod
    def distance(pos_a, pos_b):
        return round((math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)), 2)
        # pos_a = click[-2]  # from the click list [-2]
        # pos_b = click[-1]  # the most recent click/the mouse location


class Budget:
    def __init__(self):
        self.total = 1000  # can change this total budget
        self.bud_history = [self.total]

    @staticmethod
    def budget_update(remain, used):
        return round((remain - used), 2)
        # remain = bud_history [-1]
        # used = ask Map.dis_history[-1] for most recent budget used

    @staticmethod   # given loc_a & distance for loc_b to draw line
    def budget_pos(city_x, city_y, d):  # d = budget remain bud_his[-1], xy = current city loc city[-1][0], [-1][1]
        cx, cy = pg.mouse.get_pos()[0] - click[-1][0], pg.mouse.get_pos()[1] - click[-1][1]
        # current mouse position
        radians = math.atan2(cy, cx)
        # give budget line follow mouse in the correct direction
        budget_pos = int(city_x + d * math.cos(radians)), int(city_y + d * math.sin(radians))
        # print("budget_dis: " + str(Map.distance(click[-1], budget_pos)), "budget: " + str(budget.bud_history[-1]))
        # print("-------------------------------")
        return int(city_x + d * math.cos(radians)), int(city_y + d * math.sin(radians))

    @staticmethod
    def budget_undo():
        undo = round(distance_his[-1] + budget.bud_history[-1], 2)
        budget.bud_history.append(undo)
        distance_his.pop(-1)
        click.pop(-1)
        pg.draw.line(screen, WHITE, click[-2], click[-1], 3)


# those should be good order, otherwise will generate twice and don't know which to use
city_start = City()
map_1 = Map()
budget = Budget()
click.append(city_start.xy)
# a lot of places use this temporary list for update, but it's different from saving


class Data:
    def __init__(self):
        self.click_his = [(0, 0)]
        self.click_time = []    # record the click time since game started
        self.movement = []      # get.rel

    @staticmethod   # find a way to code to adding to new instance
    def saving():
        data1.click_his.append(pg.mouse.get_pos())
        data1.movement.append(pg.mouse.get_rel())
        tick_second = round((pg.time.get_ticks()/1000), 2)
        data1.click_time.append(tick_second)

    @staticmethod
    def game_update():  # update with input for new draw information
        click.append(draw_map.collision(pg.mouse.get_pos()))
        distance_his.append(map_1.distance(click[-2], click[-1]))
        map_1.dis_history.append(map_1.distance(click[-2], click[-1]))
        budget.bud_history.append(budget.budget_update(budget.bud_history[-1], map_1.dis_history[-1]))


data1 = Data()


class Draw:
    # need to code so you can't double click a city
    @staticmethod
    def collision(mouse_loc):
        for city in map_1.all_city_xy:
            x1, y1 = city[0], city[1]
            x2, y2 = mouse_loc  # pg.mouse.get_pos()
            distance = math.hypot(x1 - x2, y1 - y2)
            if distance <= 14:  # radius for each city circle and mouse circle is 7
                return x1, y1

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

    @staticmethod
    def mouse_limit(mouse_loc):
        mouse_dis = Map.distance(click[-1], mouse_loc)
        if mouse_dis < budget.bud_history[-1]:
            pg.draw.circle(screen, GREEN, pg.mouse.get_pos(), 5)
        else:
            pg.draw.circle(screen, GREEN, Budget.budget_pos(click[-1][0], click[-1][1], budget.bud_history[-1]), 5)

    @staticmethod
    def road_visual():
        pg.draw.lines(screen, BLACK, False, click, 3)
        pg.draw.line(screen, WHITE, click[0], city_start.xy, 3)
        # to hide the first line draw from origin
        pg.draw.circle(screen, RED, city_start.xy, 10)

    @staticmethod
    def instruction():
        Draw.text_write("Press Z to UNDO", 50, BLACK, 100, 200)
        pg.draw.rect(screen, WHITE, (100, 200, 1000, 50), 1)
        Draw.text_write("Press Return to SUBMIT", 50, BLACK, 100, 300)
        pg.draw.rect(screen, WHITE, (100, 300, 1000, 50), 1)
        # Draw.text_write("Press Space to the next trial", 50, BLACK, 100, 400)
        # pg.draw.rect(screen, WHITE, (100, 400, 1000, 50), 1)
        # Draw.text_write("Undo", 50, BLACK, 900, 600)
        # pg.draw.rect(screen, GREEN, (900, 600, 100, 50), 3)
        # those variable should be set at the top, so it's obvious

    @staticmethod
    def text_write(text, size, color, x, y):  # function that can display any text
        font_object = pg.font.SysFont(pg.font.get_default_font(), size)
        text_surface = font_object.render(text, True, color)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = x, y
        screen.blit(text_surface, text_rectangle.center)

    @staticmethod
    def game_end():
        pg.draw.rect(screen, BLACK, (600, 600, 600, 200), 0)
        Draw.text_write('Your score is ' + str(len(click) - 2), 60, WHITE, 650, 650)
        pg.display.flip()

    def __init__(self):
        self.road_visual()
        self.mouse_limit(pg.mouse.get_pos())
        self.instruction()
        city_visit = len(click) - 2
        # find a better structure to place this background information
        Draw.text_write("Score: " + str(city_visit), 100, BLACK, 1000, 200)
        for cities in map_1.all_city_xy:
            self.city_rect = (pg.draw.circle(screen, BLACK, cities, 7))
        pg.draw.line(screen, GREEN, click[-1], Budget.budget_pos(click[-1][0], click[-1][1], budget.bud_history[-1]), 3)
        if budget.bud_history[-1] > Map.distance(click[-1], pg.mouse.get_pos()):
            self.auto_snap(pg.mouse.get_pos())
            # this is still a bug
        pg.display.flip()
        screen.fill(WHITE)
        clock.tick(FPS)

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


# class objects
draw_map = Draw()
mouse = pg.mouse.get_pos()

# loop for displaying until quit
while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
        elif event.type == pg.MOUSEMOTION:
            draw_map.__init__()
        elif event.type == pg.MOUSEBUTTONDOWN:
            if pg.mouse.get_pressed() and event.button == 1:
                if draw_map.collision(pg.mouse.get_pos()):
                    print("collision: " + str(draw_map.collision(pg.mouse.get_pos())))
                    if budget.bud_history[-1] >= map_1.distance(click[-1], draw_map.collision(pg.mouse.get_pos())):
                        data1.game_update()
                        data1.saving()
                        draw_map.__init__()
                # else:
                    # budget.bud_history[-1] <= Map.distance(click[-1], pg.mouse.get_pos()):
                    # draw_map.refresh()
        elif event.type == pg.KEYDOWN:
            if pg.key.get_pressed() and event.key == pg.K_z:
                budget.budget_undo()
                draw_map.__init__()
            if event.key == pg.K_RETURN:
                draw_map.game_end()
                pg.time.wait(600)
            # if event.key == pg.K_SPACE:
            #     draw_map.refresh()
            if event.key == pg.K_ESCAPE:
                done = True   # very important, otherwise stuck
                pg.display.quit()


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