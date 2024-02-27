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

#env.close()