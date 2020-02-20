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
    text_write('Welcome! ',50, BLACK, 50, 300, screen)
    text_write('You are invited to be part of a research study to learn about spatial planning.', 50, BLACK, 50, 400,screen)
    text_write('This study is conducted by Dr. Wei Ji Ma, in the Department of Psychology ', 50, BLACK, 50, 500,screen)
    text_write('and the Center for Neural Science in the Faculty of Arts & Sciences', 50, BLACK, 50, 600,screen)
    text_write('at New York University.', 50, BLACK, 50, 700, screen)
    text_write('Press SPACE to continue', 50, BLACK, 50, 900, screen)

def ins_2(screen): 
    text_write('You need to complete 3 types of tasks in this study.',50, BLACK, 50, 300, screen)
    text_write('The researcher will guide you through some instructions shortly.', 50, BLACK, 50, 400,screen)
    text_write('This study takes approximately 40-50 minutes to complete.', 50, BLACK, 50, 500,screen)
    text_write('Press SPACE to continue', 50, BLACK, 50, 900, screen)

def incentive_1(screen):
    text_write('You will be compensated $12 for your participation. ',50, BLACK, 50, 300, screen)
    text_write('There will be up to $10 bonus added based on your performance in Road Construction.', 50, BLACK, 50, 400,screen)
    text_write('We will randomly select 10 scores from your Road Construction performance.', 50, BLACK, 50, 500,screen)
    text_write('Their sum will be multiplied by 0.01 as your bonus. ', 50, BLACK, 50, 600, screen)
    text_write('Press SPACE to see an example', 50, BLACK, 50, 900, screen)

def incentive_2(screen):
    text_write('For example, your 10 randomly selected scores are 64, 25, 49, 64, 36, 49, 49, 64, 100, 25',50, BLACK, 50, 300, screen)
    text_write('Your total bonus score will be 525 ', 50, BLACK, 50, 400,screen)
    text_write('You will be awarded with $5.25 in addition to your $12', 50, BLACK, 50, 500,screen)
    text_write('Thus, you will receive a total of $17.25 upon completion ', 50, BLACK, 50, 600, screen)
    text_write('Now, you are ready to start, and the researcher will leave the room.', 50, BLACK, 50, 700, screen)
    text_write('Press SPACE to start. ', 50, BLACK, 50, 900, screen)

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

def incentive_instruction(screen):    
    # instruction
    # -------------------------------------------------------------------------    
    ins = True
    
    screen.fill(GREY)
    incentive_1(screen)
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
    incentive_2(screen)
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
