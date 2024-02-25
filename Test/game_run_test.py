import gym
import pygame
from environment import TowerBuildingEnv
clock = pygame.time.Clock()

env = TowerBuildingEnv(pixel_x = 600, pixel_y = 600, goal_width = 300, goal_height = 200, grid_size = 30, max_joints = 20)


while True:
    stop = False
    action = env.action_space.sample()
    new_obs, reward, done = env.step(action)
    while stop == False or env.calculate_stability()[1] >= 0.01:
        env.world.Step(1/60, 6, 2)
        clock.tick(60)
        env.render()
        pygame.display.flip()
        
        #print(f"env.calculate_stability(): {env.calculate_stability()}")
        stop = True
    print(f"Observation: {new_obs}, Reward: {reward}, Done: {done}")
    if done:
        print(f"win!")
        break

'''
while True:
    env.world.Step(1/60, 6, 2)
    if stop == False:
        for _ in range(100):
            action = env.action_space.sample()
            new_obs, reward, done = env.step(action)
            print(f"Observation: {new_obs}, Reward: {reward}, Done: {done}")
            if done:
                env.reset()
    env.render()
    pygame.display.flip()
    print(f"env.calculate_stability(): {env.calculate_stability()}")
    clock.tick(60)
    stop = True

for _ in range(100):
    action = env.action_space.sample()
    new_obs, reward, done = env.step(action)
    env.render()
    print(f"Episode Step: {_}, Observation: {new_obs}")
    if done:
        env.reset()
'''

#env.close()