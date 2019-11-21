import pygame as pg

pg.init()
pg.font.init()

# conditions
done = False

# display setup
screen = pg.display.set_mode((2000, 1500))  # display surface
clock = pg.time.Clock()
FPS = 30  # tracking time check how to use
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
screen.fill(WHITE)
clock.tick(FPS)

# user_input = pg.event.Event(pg.KEYDOWN, unicode)
user_list = []

def text_input():
    input_rec = pg.Rect(100, 200, 1000, 50)
    pg.key.set_text_input_rect(input_rec)
    pg.key.start_text_input()


def text_write(text, size, color, x, y):  # function that can display any text
    font_object = pg.font.SysFont(pg.font.get_default_font(), size)
    text_surface = font_object.render(text, True, color)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = x, y
    screen.blit(text_surface, text_rectangle.center)

while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
        if event.type == pg.KEYDOWN:
            user = event.unicode
            text_write(user, 50, BLACK, 500, 500)
            pg.display.flip()
            user_list.append(user)
            screen.fill(WHITE)
            print(user_list)

            # if pg.key.get_pressed():
               # print(pg.event.EventType.KEYDOWN.__dict__)
        # text_input()
