import pygame as pg

# conditions
choice = True
done = True

#Helper function
def text_write(text, size, color, x, y,screen):  # function that can display any text
    font_object = pg.font.SysFont(pg.font.get_default_font(), size)
    text_surface = font_object.render(text, True, color)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = x, y
    screen.blit(text_surface, text_rectangle.center)

class ScoreBar:
    def __init__(self):
        # only call once when initiated for this part
        # score bar parameters
        self.width = 100
        self.height = 900
        self.box = 8

        # center for labels
        self.box_height = self.height / self.box
        self.center_list = []
        self.box_center(self.width, self.height)

        # incentive score indicator
        self.index = 1
        self.indicator_loc = self.center_list[self.index]

        # calculate incentive: N^2
        self.incentive()

    def box_center(self, width, height):
        for i in range(10):
            width = self.width / 2 + 1500 # larger the number, further to right
            height = self.box_height / 2
            height += (i * height)
            loc = width, height
            self.center_list.append(loc)

    def indicator(self): # call this function to updates arrow location
        self.index += 1
        self.indicator_loc = self.center_list[self.index]
        return self.index

    def incentive(self):
        self.score = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.incentive_score = []
        for i in self.score:
            i = i ** 2
            self.incentive_score.append(i)


class Draw:
    # 这个是根据屏幕来画那个方框
    def __init__(self, scorebar, screen_width, screen_height, screen):
        #bar parameters
        left = scorebar.center_list[0][0] - 20
        top = int(screen_height - scorebar.height) / 2
        width = scorebar.width
        height = scorebar.height

        # draw/label incentive number on the screen
        self.bar_shape = pg.draw.rect(screen, BLACK, (left, top, width, height), 2)  # width for line thickness
        self.number(scorebar,screen)
        self.arrow(scorebar,screen)

    def number(self, scorebar, screen):
        for i in range(10):
            loc = scorebar.center_list[i]
            text = scorebar.incentive_score[i]
            text_write(str(text), 60, BLACK, loc[0], loc[1], screen) # larger number, further to right

    def arrow(self, scorebar,screen):
        # arrow parameter
        point = (scorebar.indicator_loc[0] - 30, scorebar.indicator_loc[1] - 30)
        v2 = point[0] - 20, point[1] + 20
        v3 = point[0] - 20, point[1] + 10
        v4 = point[0] - 40, point[1] + 10
        v5 = point[0] - 40, point[1] - 10
        v6 = point[0] - 20, point[1] - 10
        v7 = point[0] - 20, point[1] - 20
        self.vertices = [point, v2, v3, v4, v5, v6, v7]
        self.arrow = pg.draw.polygon(screen, BLACK, self.vertices)


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

WIDTH = 1900
HEIGHT = 1000

pg.init()
pg.font.init()

# display setup
screen = pg.display.set_mode((WIDTH, HEIGHT), flags=pg.RESIZABLE)  # pg.FULLSCREEN pg.RESIZABLE
screen.fill(WHITE)
score1 = ScoreBar()
arrow = Draw(score1, WIDTH, HEIGHT, screen)
pg.display.flip()

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                screen.fill(WHITE)
                score1.indicator()
                arrow = Draw(score1, WIDTH, HEIGHT, screen)
                pg.display.update()