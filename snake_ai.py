#vsa

import pygame
import random
import sys
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from model import Linear_QNet, QTrainer

pygame.init()
font = pygame.font.SysFont('arial', 25)

RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Game:

    def __init__(self, width=800, height=600):

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode([self.width, self.height])
        self.x = [(width / 2)]
        self.y = [(height / 2)]
        self.snake_vel = 30
        self.x_apple = random.randrange(10, self.width - 30)
        self.y_apple = random.randrange(10, self.height - 30)
        self.reward = 0
        self.score = 0
        self.length = 1
        self.direction = [0,0,0,1]
        self.iteration = 0
        self.done = False
        pygame.display.set_caption('Snake')

    def move_left(self):
        self.direction = [1,0,0,0]

    def move_up(self):
        self.direction = [0,1,0,0]

    def move_right(self):
        self.direction = [0,0,0,1]

    def move_down(self):
        self.direction = [0,0,1,0]


    def draw(self):
        
        self.screen.fill(BLACK)
        for i in range(self.length):
            pygame.draw.rect(self.screen, WHITE,(self.x[i], self.y[i], 30, 30))

            
        if ((self.x[0] >= self.x_apple-30) and (self.x[0] <= self.x_apple +30 )) and ((self.y[0] >= self.y_apple -30 ) and (self.y[0] <= self.y_apple + 30)):

            numbers_x = [x for x in range(self.width-40) if x not in self.x]
            self.x_apple = random.choice(numbers_x)
            numbers_y = [y for y in range(30,self.height-30) if y not in self.y]
            self.y_apple = random.choice(numbers_y)
                
            self.score += 1
            self.reward = 10
            self.increase_length()
            
        pygame.draw.rect(self.screen, RED,(self.x_apple, self.y_apple, 30, 30))
                         
    def snake_move(self):

        for i in range(self.length-1,0,-1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        if self.direction == [1,0,0,0]:
            self.x[0] -= self.snake_vel
        if self.direction == [0,0,0,1]:
            self.x[0] += self.snake_vel
        if self.direction == [0,1,0,0]:
            self.y[0] += self.snake_vel
        if self.direction == [0,0,1,0]:
            self.y[0] -= self.snake_vel

        self.colision(self.x , self.y)
        self.colision_wall(self.x[0], self.y[0])
       
    def colision(self, x_snake, y_snake):
        if self.length >= 3:
            for i in range(3, self.length):
                if x_snake[0] >= x_snake[i] and x_snake[0] < x_snake[i] + self.snake_vel:
                    if y_snake[0] >= y_snake[i] and y_snake[0] < y_snake[i] + self.snake_vel:                   
                        self.done = True
                        self.reward = -100        

        return self.done

                        
    def colision_wall(self, x_snake, y_snake):    

        if x_snake < -20 or x_snake >= self.width :
           self.done = True
           self.reward = -100

        if y_snake < -20 or y_snake >= self.height :
           self.done = True
           self.reward = -100

        return self.done
           
  
    def reset_game(self):

        self.x = [(self.width / 2)]
        self.y = [(self.height / 2)]
        self.x_apple = random.randrange(10, self.width - 30)
        self.y_apple = random.randrange(10, self.height - 30)
        self.length = 1
        self.score = 0
        self.reward = 0
    
    def increase_length(self):
        self.length += 1
        self.x.append(self.x[-1])
        self.y.append(self.y[-1])

    def game_steps(self, action):
        
            #left
        if np.array_equal(action,[1,0,0,0]) and self.direction != [0,0,0,1]:
            self.move_left()
            #right
        if np.array_equal(action,[0,0,0,1]) and self.direction != [1,0,0,0]:
            self.move_right()
            #down
        if np.array_equal(action,[0,1,0,0]) and self.direction != [0,1,0,0]:
            self.move_down()
            #up
        if np.array_equal(action,[0,0,1,0]) and self.direction != [0,0,1,0]:
            self.move_up()

        return self.done, self.score, self.reward


import numpy as np
BATCH_SIZE = 1000

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(17, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, reward, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, reward, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_state(self, gameplay):
        x_snake = gameplay.x[0]
        y_snake = gameplay.y[0]
        
        turn_left = gameplay.direction == [1,0,0,0]
        turn_right = gameplay.direction == [0,0,0,1]
        turn_up = gameplay.direction == [0,1,0,0]
        turn_down = gameplay.direction == [0,0,1,0]

        hit_snake_left = [x-30 for x in gameplay.x.copy()]
        hit_snake_right = [x+30 for x in gameplay.x.copy()]
        hit_snake_down = [y-30 for y in gameplay.y.copy()]
        hit_snake_up = [y+30 for y in gameplay.y.copy()]

        states=[
        gameplay.x_apple < gameplay.x[0], #food left
        gameplay.x_apple > gameplay.x[0], #food right
        gameplay.y_apple < gameplay.y[0], #food up
        gameplay.y_apple > gameplay.y[0], #food down
        
        turn_left,
        turn_right,
        turn_up,
        turn_down,

        #danger straight
        (turn_up and gameplay.colision_wall(x_snake, (y_snake - 30))),

        #danger down
        (turn_down and gameplay.colision_wall(x_snake, (y_snake + 30))),
        
        #danger right
        (turn_right and gameplay.colision_wall((x_snake + 30), y_snake)),

        #danger left
        (turn_left and gameplay.colision_wall((x_snake - 30), y_snake)),

        #hit the snake left
        (turn_left and gameplay.colision(hit_snake_left ,gameplay.y)),

        #hit the snake right
        (turn_right and gameplay.colision(hit_snake_right ,gameplay.y)),

        #hit the snake down 
        (turn_down and gameplay.colision(gameplay.x , hit_snake_down)),

        #hit the snake up
        (turn_up and gameplay.colision(gameplay.x , hit_snake_up)),
        
        gameplay.colision(gameplay.x, gameplay.y)
        ]
        return np.array(states, dtype=int)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.n_games
        final_move = [0,0,0,0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def main():
    
    main_font = pygame.font.SysFont("comicsans", 30)
    clock = pygame.time.Clock()
    n_games = 0
    gameplay = Game(600,400)
    agent = Agent()
    plot_scores = []
    plot_mean_scores = []
    looping_list_x = []
    looping_list_y = []
    total_score = 0
    record = 0
    FPS = 10
    run = True
    iteration = 0
    length_list_x = []
    length_list_y = []
    
    def redraw_window():
        pygame.display.update()
        gameplay.draw()
        score_label = main_font.render(f"Score: {gameplay.score}", True, (255, 255, 255))

        gameplay.screen.blit(score_label, (10, 10))

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w    

      
    while run:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        clock.tick(FPS)
        redraw_window()
        state_old = agent.get_state(gameplay)
        final_move = agent.get_action(state_old)
        done, score, reward = gameplay.game_steps(final_move)
        gameplay.snake_move()
        state_new = agent.get_state(gameplay)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        length_list_x.append(gameplay.x.copy())
        length_list_y.append(gameplay.y.copy())
        
        if len(length_list_x) > 9 and len(length_list_y) > 9 :

            if length_list_x[-1] == length_list_x[-5] and length_list_x[-2] == length_list_x[-6] and length_list_x[-3] == length_list_x[-7] and length_list_x[-4] == length_list_x[-8] and length_list_x[-5] == length_list_x[-9]:
                if length_list_y[-1] == length_list_y[-5] and length_list_y[-2] == length_list_y[-6] and length_list_y[-3] == length_list_y[-7] and length_list_y[-4] == length_list_y[-8] and length_list_y[-5] == length_list_y[-9]:
                    gameplay.reward = -100
                    gameplay.reset_game()
                    length_list_x = []
                    length_list_y = []
                    
                    agent.n_games += 1
                    agent.train_long_memory()
                    if score > record:
                        record = score
                        agent.model.save()


                    print(f'Game {agent.n_games} Score {score} Record: {record}')

                    plot_scores.append(score)
                    mean_score = total_score / agent.n_games
                    plot_mean_scores.append(mean_score)
                 
                    gameplay.done = False
                    
                
        if done:
            length_list_x = []
            length_list_y = []
            gameplay.reset_game()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()


            print(f'Game {agent.n_games} Score {score} Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
        
            gameplay.done = False
    plt.plot(moving_average(plot_scores, 10))
    plt.plot(plot_mean_scores)
    plt.tight_layout()
    plt.show()
          
            
if __name__ == '__main__':
    main()
    pygame.quit()
    sys.exit()
