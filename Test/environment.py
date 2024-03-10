import gym
import pygame
from Box2D import (b2World, b2PolygonShape, b2CircleShape, b2_dynamicBody, b2DistanceJointDef, b2WeldJointDef, b2, b2FixtureDef)
import random
import numpy as np
import os
import platform
from PIL import Image
env = gym.make('CartPole-v0')

import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class TowerBuildingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, screen_x, screen_y, goal_width, goal_height, grid_size, max_joints):
        # Pygame Initialization
        pygame.init()

        # Box2D World Initialization
        self.world = b2World(gravity=(0, -10), doSleep=True) # Box2D world
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.screen = pygame.display.set_mode((screen_x, screen_y))
        self.goal_width = goal_width
        self.goal_height = goal_height
        self.max_joints = max_joints
        self.grid_size = grid_size
        self.cell_size = (screen_x // grid_size, screen_y // grid_size) # (20, 20) pixels per grid cell
        self.clock = pygame.time.Clock()
        self.block_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]
        self.ppm = 20.0  # pixels per meter
        self.ground = self.world.CreateStaticBody(position=(0, 0))
        self.ground.CreateEdgeFixture(vertices=[(-1500, 0), (1500, 0)], density=1, friction=0.3)

        self.tower_grid = np.zeros((goal_width, goal_height))

        # Image index
        self.image_index = 0

        # Coordinates calculation helper parameters
        self.is_valid = False
        self.is_valid_close = False
        self.closest_squared = 10000.0 # Set the closest squared to a value that yields a zero reward
        self.block_radius = self.cell_size[0] * 0.5 * np.sqrt(5) # 22.3 pixels
        self.max_vicinity_squared = 5000
        
        self.max_h_coord = 0
        self.min_x_coord = screen_x/2
        self.max_x_coord = screen_x/2
        
        # State variables
        self.width = 0
        self.height = 0
        self.blocks = []
        self.current_score = 0
        self.steps = 0
        self.records = [] # (step, score, width, height)

        #self.action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (2,))
        #self.action_space = gym.spaces.Box(low = np.array((0,0)), high = np.array((screen_x, screen_y)), shape = (2,))

        # Score function parameters
        ## width progress = sigma * (goal_width ^ alpha - (width - goal_width) ^ alpha), alpha > 1, sigma > 0
        ## height reward = beta * height ^ theta, thea > 1
        ## closeness progress = -(omega * closest_squared)^zeta + phi
        ## stability punishment = kappa * (average_speed + max_speed)
        ## efficiency punishment = delta * block_num
        ## validity punishment = mu, mu < 0
        self.alpha = 1.6
        self.sigma = 0.0001
        self.beta = 0.0002
        self.theta = 2
        self.omega = 0.00005
        self.zeta = 1.001
        self.phi = 1
        self.kappa = -0.01
        self.delta = -0.005
        self.mu = -10.0

        # Place the first block
        self.place_block(screen_x/2, screen_y/2)

    def place_block(self, x_coord, y_coord, angle = None):
        if angle is None:
            angle = random.uniform(0, 180) * (np.pi / 180)
        else:
            angle = angle * (np.pi / 180)
        
        new_body = self.world.CreateDynamicBody(
            position = (x_coord/self.ppm, y_coord/self.ppm),
            #angle = random.choice([0, 45, 90]) * (np.pi / 180),
            angle = angle,
        )
        new_block = new_body.CreatePolygonFixture(box = (self.cell_size[0]/self.ppm, self.cell_size[1]/2/self.ppm), density = 1, friction = 0.3)
        self.blocks.append(new_block)
        self.new_block = new_block

    def step(self, action): # Return true if a valid and close placement is found
        # Randomly choose a valid and close point
        self.is_valid = False
        self.is_valid_close = False

        x_coord, y_coord, angle = action[0], action[1], action[2]

        ### Attempt to find a valid and close placement within 1 single loop
        '''
        while True:
            x_coord = self.screen_x * random.random()
            y_coord = self.screen_y * random.random()
            self.is_valid, closest_squared = self.is_valid_placement(x_coord, y_coord)
            if self.is_valid and self.is_close_enough(closest_squared, self.max_vicinity_squared):
                block_x_coord, block_y_coord = x_coord, y_coord
                self.is_valid_close = True
                break
        '''

        ### Attempt 1 placement per loop, if valid and close placement is found place a block
        '''
        self.is_valid, self.closest_squared = self.is_valid_placement(x_coord, y_coord)
        if self.is_valid and self.is_close_enough(self.closest_squared, self.max_vicinity_squared):
            block_x_coord, block_y_coord = x_coord, y_coord
            self.is_valid_close = True

        if self.is_valid_close:
            self.place_block(block_x_coord, block_y_coord, angle)
        else:
            self.closest_squared = 1000000.0
        '''

        ### Attempt 1 placement per loop, if valid placement is found place a block
        self.is_valid, self.closest_squared = self.is_valid_placement(x_coord, y_coord)
        if self.is_valid:
            block_x_coord, block_y_coord = x_coord, y_coord
            self.place_block(block_x_coord, block_y_coord, angle)
        else:
            self.closest_squared = 10000.0 # Set the closest squared to a value that yields a zero reward


    def is_valid_placement(self, grid_x_coord, grid_y_coord):
        # Create a temporary fixture representing the potential new block
        potential_block_body = self.world.CreateDynamicBody(
            position=(grid_x_coord/self.ppm, grid_y_coord/self.ppm)
        )
        
        # Create a temporary circular fixture attached to the potential block body
        p_f = potential_block_body.CreateCircleFixture(radius=50/self.ppm, density=1, friction=0.3)

        closest_squared = 360000.0
        for e in self.blocks:
            if self.fixtures_overlap(p_f, e):
                self.world.DestroyBody(potential_block_body)
                return False, None
            else:
                block_x_coord = e.body.position.x * self.ppm
                block_y_coord = e.body.position.y * self.ppm
                #block_x_coord, block_y_coord = self.grid_to_world_coords(e.body.position.x, e.body.position.y)
                dist_squared = (block_x_coord - grid_x_coord) ** 2 + (block_y_coord - grid_y_coord) ** 2
                closest_squared =  min(dist_squared, closest_squared)
        
        self.world.DestroyBody(potential_block_body)
        return True, closest_squared

    def is_close_enough(self, dist_squared, max_dist_squared):
        return dist_squared < max_dist_squared

    def fixtures_overlap(self, fixture1, fixture2):
        # ... Check if fixture1 and fixture2 overlap ...
        overlap = b2.testOverlap(fixture1.shape, 0, fixture2.shape, 0, fixture1.body.transform, fixture2.body.transform)
        return overlap

    def get_random_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r, g, b)
    
    def grid_to_world_coords(self, grid_x, grid_y):
        return grid_x * self.cell_size[0] + self.cell_size[0] // 2, grid_y * self.cell_size[1] + self.cell_size[1] // 2
 
    def update_records(self): # return the latest step, score, width, and height
        #1 Update the step count
        self.steps += 1

        #2 Update the width and height
        self.max_h_coord = 0
        self.min_x_coord = self.screen_x/2
        self.max_x_coord = self.screen_x/2
        for block in self.blocks:
            for vertex in block.shape.vertices:
                x_coord, y_coord = block.body.transform * vertex * self.ppm
                self.max_h_coord = max(self.max_h_coord, y_coord)
                self.min_x_coord = min(self.min_x_coord, x_coord)
                self.max_x_coord = max(self.max_x_coord, x_coord)

        self.width = (self.max_x_coord - self.min_x_coord)
        self.height = self.max_h_coord

        #3 Update the score
        w_h_progress = self.calculate_progress()
        closeness_progress = self.calculate_closeness()
        stability_punishment = self.calculate_stability()[2]
        efficiency_punishment = self.calculate_efficiency()
        if self.is_valid:
            validity_punishment = 0
        else:
            validity_punishment = self.mu
            #validity_punishment = 0
        print(f"Progress: {w_h_progress:.4f}, Closeness: {closeness_progress:.4f}, Stability: {stability_punishment:.4f}, Efficiency: {efficiency_punishment:.4f}, Validity: {validity_punishment:.4f}")
        self.current_score = w_h_progress + closeness_progress + stability_punishment + efficiency_punishment + validity_punishment

        #4 Record the step, score, width, height, and validity
        self.records.append((self.steps, self.current_score, self.width, self.height, self.is_valid))
        return (self.steps, self.current_score, self.width, self.height, self.is_valid)

    def calculate_progress(self):
        progress_x = self.sigma * (self.goal_width ** self.alpha - abs(self.width - self.goal_width) ** self.alpha)
        progress_y = self.beta * (self.height ** self.theta)
        return progress_x + progress_y
    
    def calculate_stability(self):
        average_speed = 0
        max_speed = 0
        
        for block in self.blocks:
            speed = block.body.linearVelocity.length
            average_speed += speed
            max_speed = max(max_speed, speed)

        average_speed /= len(self.blocks)
        punishment = self.kappa * (average_speed + max_speed)
        return [average_speed, max_speed, punishment]
    
    def calculate_efficiency(self):
        return self.delta * len(self.blocks)
    
    def calculate_closeness(self):
        return - (self.omega * self.closest_squared) ** self.zeta + self.phi
    
    def check_done(self):
        # Did the tower reach the goal? Did it collapse? etc...
        if self.check_win():
            import matplotlib.pyplot as plt

            records = self.records
            x = [entry[0] for entry in records]
            y = [entry[1] for entry in records]
            w = [entry[2] for entry in records]
            h = [entry[3] for entry in records]

            fig, ax1 = plt.subplots(figsize=(16, 9))

            # Plot y vs x
            ax1.plot(x, y, 'b-', label='Score vs Steps')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Score', color='b')
            ax1.tick_params('y', colors='b')

            # Create a second y-axis
            ax2 = ax1.twinx()

            # Plot w vs x
            ax2.plot(x, w, 'r-', label='Width vs Steps')
            ax2.set_ylabel('Width', color='r')
            ax2.tick_params('y', colors='r')

            # Create a third y-axis
            ax3 = ax1.twinx()

            # Offset the third y-axis
            ax3.spines['right'].set_position(('outward', 60))

            # Plot h vs x
            ax3.plot(x, h, 'g-', label='Height vs Steps')
            ax3.set_ylabel('Height', color='g')
            ax3.tick_params('y', colors='g')

            final_step = self.steps
            final_score = self.current_score
            # Add a title
            plt.title(f'Tower Building Progress\nFinal Steps: {final_step}\nFinal Score: {final_score}')

            # Adjust the plot layout
            fig.tight_layout()

            plt.savefig('score_plot.png')
            os.system('xdg-open score_plot.png')

            return True
        else:
            return False
    
    def check_win(self):
        # Did the tower reach the goal?
        # ... Compare 'self.tower_grid' (or your representation) to 'self.target_tower'...
        win = False
        if self.width >= self.goal_width and self.height >= self.goal_height:
        #if height >= self.goal_height:
            win = True
        return win
    
    def get_screen(self):
        raw_screen = pygame.surfarray.array3d(self.screen)

        gray_image = Image.fromarray(raw_screen).convert('L')
        gray_image.save(f'screenshot_{self.image_index}.png')
        resized_screen = gray_image.resize((84, 84), Image.BILINEAR)
        # Save resized gray image
        resized_screen.save(f'screenshot_resized_{self.image_index}.png')
        #resized_screen.save(f'screenshot_resized_{self.image_index}.png')

        resized_screen = np.array(resized_screen, dtype=np.uint8)/255.0
        # export the raw_screen to a file
        #pygame.image.save(raw_screen, f'screenshot_{self.image_index}.png')
        self.image_index += 1

        return resized_screen

    def reset(self):
        # Coordinates calculation helper parameters
        self.max_h_coord = 0
        self.min_x_coord = self.screen_x/2
        self.max_x_coord = self.screen_x/2
        
        # State variables
        self.width = 0
        self.height = 0
        self.blocks = []
        self.steps = 0
        self.current_score = 0
        self.records = [] # (step, score, width, height)

        # Image index
        self.image_index = 0

        # Place the first block
        self.place_block(self.screen_x/2, self.screen_y/2)
    
    def render(self):
        # Clear the screen (Example: Fill with white)
        self.screen.fill((255, 255, 255))

        # Example: Draw the ground plane
        ground_y = self.ground.position.y  # Assuming the ground is at y=0
        #pygame.draw.rect(self.screen, (100, 100, 100), (0, ground_y, self.screen_x, self.screen_y - ground_y))

        # Draw blocks 
        for i, block in enumerate(self.blocks):
            # draw colorful blocks
            #self.draw_block(block.body, self.block_colors[i % len(self.block_colors)])
            # draw grey blocks
            self.draw_block(block.body, (50, 50, 50))
            # draw block outlines            
            
    def draw_block(self, body, color):
        for fixture in body.fixtures:
            shape = fixture.shape
            vertices = [(body.transform * v) * self.ppm for v in shape.vertices]
            vertices = [(v[0], 600 - v[1]) for v in vertices]
            pygame.draw.polygon(self.screen, color, vertices)


    def close(self):
        pygame.quit()
