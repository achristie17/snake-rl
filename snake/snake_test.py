from snake_game import Snake
import pygame
import time

game = Snake(10)
pygame.init()
while True:
    done =  False
    score = 0
    for event in pygame.event.get():
        # checking if keydown event happened or not
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                score, done = game.step(0)
            elif event.key == pygame.K_RIGHT:
                score, done = game.step(1)
            elif event.key == pygame.K_UP:
                score, done = game.step(2)
            elif event.key == pygame.K_DOWN:
                score, done = game.step(3)
    print(game)
    if done:
        break
    time.sleep(0.05)