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

def text_input():
    input_box = pg.Rect(100, 100, 140, 32)
    input_rec = pg.Rect(100, 200, 1000, 50)
    pg.key.set_text_input_rect(input_rec)
    pg.key.start_text_input()
    print(pg.key.start_text_input())

while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
        if event.type == pg.KEYDOWN:
            if pg.key.get_pressed():
               print(pg.event.EventType.KEYDOWN.__dict__)
        # text_input()