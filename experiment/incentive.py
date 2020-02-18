import pygame as pg

# conditions
choice = True
done = True

#Helper function
def text_write(text, size, color, x, y, screen):  # function that can display any text
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
        self.height = 500
        self.box = 8

        # center for labels
        self.box_center(self.width, self.height)

        # incentive score indicator
        self.index = 0
        self.indicator_loc = self.center_list[self.index]

        # calculate incentive: N^2
        self.incentive()

    def box_center(self, width, height):
        self.box_height = self.height / self.box
        self.center_list = []
        self.uni_height = self.box_height / 2
        self.x = self.width / 2 + 1500 # larger the number, further to right

        for i in range(10):
            y =  i * self.box_height + self.uni_height
            loc = self.x, y
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
    def __init__(self, scorebar, screen):
        #bar parameters
        self.top = 200
        width = scorebar.width
        height = scorebar.height

        # draw/label incentive number on the screen
        self.number(scorebar,screen)
        self.arrow(scorebar,screen)

    def number(self, scorebar, screen):
        left = scorebar.center_list[0][0] - 25
        for i in range(10):
            loc = scorebar.center_list[i]
            text = scorebar.incentive_score[i]
            text_write(str(text), int(scorebar.box_height - 15), BLACK, loc[0], loc[1]+self.top , screen) # larger number, further to right
            pg.draw.rect(screen, BLACK, (left, loc[1]+self.top-scorebar.uni_height, 
                                         scorebar.width, scorebar.box_height), 2)  # width for line thickness
    def arrow(self, scorebar,screen):
        # arrow parameter
        point = (scorebar.indicator_loc[0] - 30, scorebar.indicator_loc[1]+self.top+10)
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
HEIGHT = 1600

if __name__ == "__main__":

    pg.init()
    pg.font.init()
    
    # display setup
    screen = pg.display.set_mode((WIDTH, HEIGHT), flags=pg.FULLSCREEN)  # pg.FULLSCREEN pg.RESIZABLE
    screen.fill(WHITE)
    score1 = ScoreBar()
    arrow = Draw(score1, screen)
    pg.display.flip()
    
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    screen.fill(WHITE)
                    score1.indicator()
                    arrow = Draw(score1, screen)
                    pg.display.update()
                if event.key == pg.K_ESCAPE:
                        pg.quit()
