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
    text_write('This study takes approximately 40-50 minutes to complete.',50, BLACK, 50, 300, screen)
    text_write('It consists of 6 parts, and each part takes about 6-8 minutes.', 50, BLACK, 50, 400,screen)
    text_write('You can take a short break after each part.', 50, BLACK, 50, 500,screen)
    text_write('The researcher will now guide you through some instructions. ', 50, BLACK, 50, 600,screen)
    text_write('Press SPACE to continue', 50, BLACK, 50, 900, screen)

def ins_2(screen): 
    text_write('In each part you will be doing one of the following 3 tasks:',50, BLACK, 50, 300, screen)
    text_write('Number Estimation, Road Construction, and Road Construction with Undo.', 50, BLACK, 50, 400,screen)
    text_write('Now we are going to show you the instructions for these 3 tasks. ', 50, BLACK, 50, 500,screen)
    text_write('Press SPACE to continue', 50, BLACK, 50, 900, screen)

def incentive_1(screen):
    text_write('You will receive $12 for your participation. ',50, BLACK, 50, 300, screen)
    text_write('There will be up to $10 bonus added based on your performance.', 50, BLACK, 50, 400,screen)
    text_write('We will randomly select 3 scores from your Road Construction with/without Undo scores as the bonus.', 50, BLACK, 50, 500,screen)
    text_write('Press SPACE to continue', 50, BLACK, 50, 900, screen)

def ins_end(screen):
    text_write('Do you have any questions? ',50, BLACK, 50, 300, screen)
    text_write('If not, press SPACE to start the experiment.', 50, BLACK, 50, 900, screen) 

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
