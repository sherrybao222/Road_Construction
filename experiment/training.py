import pygame as pg
from random import randrange

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
    text_write('Welcome. This study takes approximately 40-60 minutes to complete.',50, BLACK, 50, 300, screen)
    text_write('It consists of 2 experiments: Road Construction and Number Estimation', 50, BLACK, 50, 400,screen)
    text_write('There are 4 parts for each experiment, and each part takes about 5-7 minutes.', 50, BLACK, 50, 500,screen)
    text_write('You can take a short break after finishing the first experiment.', 50, BLACK, 50, 600,screen)
    text_write('Press RETURN to continue', 50, BLACK, 50, 900, screen)


def incentive_1(screen):
    text_write('You will receive $12 for your participation. ',50, BLACK, 50, 300, screen)
    text_write('There will be up to a $5 bonus added based on your performance in Road Construction.', 50, BLACK, 50, 400,screen)
    text_write('We will randomly select 2 results from Road Construction and Road Construction with Undo to calculate your bonus.', 50, BLACK, 50, 500, screen)
    text_write('In order to maximize your bonus, it’s important to try your best in each trial.', 50, BLACK, 50, 600, screen)
    text_write('Press RETURN to continue', 50, BLACK, 50, 700, screen)

def ins_2(screen): 
    text_write('Once you finish Road Construction, you will continue to read the instruction for Experiment Two: Number Estimation.',50, BLACK, 50, 300, screen)
    text_write('Feel free to reach out to the researcher at any point during the experiment if you have questions.', 50, BLACK, 50, 400,screen)
    text_write('If not, press RETURN to start Road Construction.', 50, BLACK, 50, 900, screen)

#def ins_end(screen):
#    text_write('Do you have any questions? ',50, BLACK, 50, 300, screen)
#    text_write('If not, press RETURN to start the experiment.', 50, BLACK, 50, 900, screen)

def exp_end(screen,ind_list_2,pay_list_2,ind_list_3,pay_list_3):
    text_write('Thank you for participating in this study.',50, BLACK, 50, 300, screen)
    text_write('The chosen trials for road construction are '+str(ind_list_2),50, BLACK, 50, 400, screen)
    text_write('Your scores are '+str([ x**2 for x in pay_list_2]),50, BLACK, 50, 500, screen)
    text_write('The chosen trials for road construction with undo are '+str(ind_list_3),50, BLACK, 50, 600, screen)
    text_write('Your scores are '+str([ x**2 for x in pay_list_3]),50, BLACK, 50, 700, screen)
    text_write('Your total payment is '+ str((sum([ x**2 for x in pay_list_2])+sum([ x**2 for x in pay_list_3]))/100+12),50, BLACK, 50, 800, screen)
    text_write('You can now notify the researcher, and you will complete a short survey.',50, BLACK, 50, 900, screen)

def payment(my_order,cond,trials,n_trl):
    ind_rc = [i for i, x in enumerate(my_order) if x == cond]
    ind_map = list(range(n_trl*ind_rc[0],n_trl*(ind_rc[0]+1)))
    ind_map.extend(list(range(n_trl*ind_rc[1],n_trl*(ind_rc[1]+1))))
    ans_list = []
    pay_list = []
    ind_list = []
    for ind in ind_map:
        ans_list.append(int(trials[ind].n_city[-1]))
    for i in range(0,2):
        ind_list.append(randrange(len(ans_list)))
        pay_list.append(ans_list.pop(ind_list[-1]))
        i = i + 1
    return ind_list,pay_list

    
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
                if event.key == pg.K_RETURN:
                    ins = False 
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()   

    ins = True


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
                if event.key == pg.K_RETURN:
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
                if event.key == pg.K_RETURN:
                    ins = False 
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()   

def end_instruction(screen,ind_list_2,pay_list_2,ind_list_3,pay_list_3):    
    # instruction
    # -------------------------------------------------------------------------    
    ins = True
    
    screen.fill(GREY)
    exp_end(screen,ind_list_2,pay_list_2,ind_list_3,pay_list_3)
    pg.display.flip()  

    while ins:
        events = pg.event.get()
        for event in events:
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
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
