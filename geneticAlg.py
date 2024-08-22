import random
from copy import deepcopy
import torch
import numpy as np
from snake.snake_game import Snake
import random
from train import PolicyNetwork

class GeneticPolicyAgent:
    def __init__(self, model, gamma=0.8):
        self.model = model
        self.gamma = gamma  # Discount rate
        self.fitness = 0    # Fitness score of the agent

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            prediction = self.model(state_tensor.unsqueeze(0))
        action = torch.argmax(prediction).item()
        return action

    def evaluate_fitness(self, game_score, num_episodes):
        # print("Avg game score: ", game_score / num_episodes)
        self.fitness = game_score / num_episodes

    def crossover(self, other_agent):
        child_model = deepcopy(self.model)
        for child_param, other_param in zip(child_model.parameters(), other_agent.model.parameters()):
            if len(child_param.data.size()) == 1:  # weights of a linear layer
                num_elements = child_param.data.size()[0]
                split = random.randint(0, num_elements)
                child_param.data[split:] = other_param.data[split:]
        
        return GeneticPolicyAgent(child_model, gamma=(self.gamma + other_agent.gamma) / 2)

    def mutate(self, mutation_rate=0.05):
        for param in self.model.parameters():
            tensor_size = param.data.size()
            for index in np.ndindex(*tensor_size):
                if random.random() < mutation_rate:
                    param.data[index] += torch.randn(1).item()


def train_genetic_algorithm(num_episodes=2, max_steps=100, board_size=10, population_size=100, generations=1000):
    game = Snake(board_size)
    # Initialize population
    population = [GeneticPolicyAgent(PolicyNetwork(12, 4)) for _ in range(population_size)]
    record = 0
    for generation in range(generations):
        # Evaluate fitness for each agent
        for agent in population:
            total_reward = 0
            if generation % 100 == 0:
                num_episodes = 10
            else:
                num_episodes = 2

            for episode in range(num_episodes):
                game = Snake(board_size)
                state = game.get_state()
                done = False
                num_steps_without_reward = 0
                game_reward = 0
                while not done:
                    action = agent.get_action(state)
                    state, reward, done = game.step(action)
                    total_reward += reward
                    game_reward += reward

                    if reward > 0:
                        num_steps_without_reward = 0
                    else: 
                        num_steps_without_reward += 1
                    
                    if num_steps_without_reward >= max_steps:
                        total_reward -= 10
                        done = True
                    # if (generation == generations-1):
                    #     print(game) 
                    #     time.sleep(0.1)
                    if game_reward > ((record *100) - 10):
                        record = (game_reward) / 100

            agent.evaluate_fitness(total_reward, num_episodes)
                

        # Sort agents by fitness
        population.sort(key=lambda agent: agent.fitness, reverse=True)
        if generation % 10 == 0:
            torch.save(population[0].model, f'policy_model_basic.pth-{generation}')
        # Select the top performers
        top_performers = population[:population_size // 2]

        # Logging average fitness
        avg_fitness = sum(agent.fitness for agent in top_performers) / (population_size // 2)
        max_fitness = population[0].fitness
        print(f"Generation {generation + 1}: Average Fitness = {avg_fitness}: Max Fitness = {max_fitness}: Record = {record}")

        # Crossover and mutation to create a new generation
        new_generation = []
        while len(new_generation) < population_size:
            parent1 = random.choice(top_performers)
            parent2 = random.choice(top_performers)
            child_agent = parent1.crossover(parent2)
            child_agent.mutate()
            new_generation.append(child_agent)

        population = new_generation



    return population


