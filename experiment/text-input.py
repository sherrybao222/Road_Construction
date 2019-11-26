#!/usr/bin/python3
import pygame_textinput
import pygame
pygame.init()

# Create TextInput-object
textinput = pygame_textinput.TextInput()

screen = pygame.display.set_mode((1000, 500))
text_screen = pygame.display.set_mode((500, 500))
clock = pygame.time.Clock()

while True:
    text_screen.fill((225, 225, 225))

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            exit()

    # Feed it with events every frame
    textinput.update(events)
    # Blit its surface onto the screen
    screen.blit(textinput.get_surface(), (50, 50))

    pygame.display.update()
    clock.tick(30)
