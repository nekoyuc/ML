import pygame
from environment import TowerBuildingEnv
import datetime
from utils import ReplayBuffer
import random

replay_buffer = ReplayBuffer(10000)

# Create the environment
# Screen_x, screen_y, goal_width, goal_height, grid_size, max_joints, replay_buffer
env = TowerBuildingEnv(600, 600, 300, 250, 30, 20)

while True:
    stop = False
    #action = env.action_space.sample()
    #env.step(action)
    while (stop == False or env.calculate_stability()[1] >= 0.005):
        env.world.Step(1/60, 6, 2)
        env.clock.tick(5000)
        env.render()
        pygame.display.flip()
        stop = True
        
    step, score, width, height, validity = env.update_records() # return the latest step, score, width, and height
    # print current time
    #current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #print(f"Current time: {current_time}")

    if validity:
        env.get_screen()
        done = env.check_done()
        '''
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current time: {current_time}")
        '''
        print(f"Width: {width}, Height: {height}")

        '''
        progress = env.calculate_progress() # Reward from tower width and height
        stability = env.calculate_stability()[2]
        efficiency = env.calculate_efficiency()
        print(f"Progress: {progress:.4f}, Stability: {stability:.4f}, Efficiency: {efficiency:.4f}, Score: {score:.4f}\n")
        '''
        if done:
            print(f"win!")
            break
    action = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 180))
    env.step(action)

#env.close()