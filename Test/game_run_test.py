import gym
import pygame
from environment import TowerBuildingEnv
import datetime
clock = pygame.time.Clock()

env = TowerBuildingEnv(screen_x = 600, screen_y = 600, goal_width = 300, goal_height = 250, grid_size = 30, max_joints = 20)


while True:
    stop = False
    action = env.action_space.sample()
    env.step(action)
    while stop == False or env.calculate_stability()[1] >= 0.01:
        env.world.Step(1/60, 6, 2)
        clock.tick(60)
        env.render()
        pygame.display.flip()
        
        #print(f"env.calculate_stability(): {env.calculate_stability()}")
        stop = True
    new_obs = env.get_observation()
    reward = env.calcualte_reward()
    done = env.check_done()
    # print current time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current time: {current_time}")
    print(f"Observation: {new_obs}, Reward: {reward}, Done: {done}\n")
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