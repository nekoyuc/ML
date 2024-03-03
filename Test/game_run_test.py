import pygame
from environment import TowerBuildingEnv
import datetime

env = TowerBuildingEnv(screen_x = 600, screen_y = 600, goal_width = 300, goal_height = 250, grid_size = 30, max_joints = 20)

while True:
    stop = False
    action = env.action_space.sample()
    #env.step(action)
    while stop == False or env.calculate_stability()[1] >= 0.01:
        env.world.Step(1/60, 6, 2)
        env.clock.tick(5000)
        env.render()
        pygame.display.flip()
        stop = True
        
    env.update_score()
    env.update_records()
    env.get_screen()

    reward = env.current_score
    done = env.check_done()

    
    
    # print current time
    #current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #print(f"Current time: {current_time}")
    print(f"Observation: width is {env.width:.4f}, height is {env.height:.4f}, Reward: {reward:.4f}\n")

    progress = env.calculate_progress()
    stability = env.calculate_stability()[2]
    efficiency = env.calculate_efficiency()
    print(f"Progress: {progress:.4f}, Stability: {stability:.4f}, Efficiency: {efficiency:.4f}\n")
    if done:
        print(f"win!")
        break

    env.step(action)

#env.close()