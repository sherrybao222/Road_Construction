import pygame as pg

# helper function
# =============================================================================
def text_write(text, size, color, x, y,screen):  # function that can display any text
    font_object = pg.font.SysFont(pg.font.get_default_font(), size)
    text_surface = font_object.render(text, True, color)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = x, y
    screen.blit(text_surface, text_rectangle.center)

# instruction
# =============================================================================
def ins_1(screen): 
    text_write('You are invited to take part in a research study',90, BLACK, 50, 300, screen)
    text_write('to learn more about spatial planning in humans.', 90, BLACK, 50, 400,screen)
    text_write('This study is conducted by Dr. Wei Ji Ma,', 90, BLACK, 50, 500,screen)
    text_write('in the Department of Psychology and the Center', 90, BLACK, 50, 600,screen)
    text_write('for Neural Science in the Faculty of Arts & Sciences', 90, BLACK, 50, 700, screen)
    text_write('at New York University.', 90, BLACK, 50, 800, screen)

def ins_2(screen): 
    text_write('There are 3 types of tasks for you to complete in this study.',90, BLACK, 50, 300, screen)
    text_write('The researcher will guide you through instructions of', 90, BLACK, 50, 400,screen)
    text_write('these 3 tasks shortly.', 90, BLACK, 50, 500,screen)
    text_write('This study takes approximately 40-50 minutes to complete,', 90, BLACK, 50, 600,screen)
    text_write('and you will be reminded when you are halfway done.', 90, BLACK, 50, 700, screen)

def training(screen):    
    # instruction
    # -------------------------------------------------------------------------    
    ins = True
    
    screen.fill(GREY)
    ins_1(screen)
    pg.display.flip()  

    while ins:
        events = pg.event.get()
        for event in events:
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    ins = False 
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()   

    ins = True

    screen.fill(GREY)
    ins_2(screen)
    pg.display.flip()  
    
    while ins:
        events = pg.event.get()
        for event in events:
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    ins = False 
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()   


# main
# =============================================================================
# setting up window, basic features 
GREY = (222, 222, 222)    
RED = (255, 102, 102)
GREEN = (0, 204, 102)
BLACK = (0, 0, 0)    

if __name__ == "__main__":
    pg.init()
    pg.font.init()
    
    # display setup
    screen = pg.display.set_mode((2000, 1600), flags=pg.FULLSCREEN)  # pg.FULLSCREEN pg.RESIZABLE

    screen.fill(GREY)
        
    training(screen)

    pg.quit()
