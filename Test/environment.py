import gym
import pygame
from Box2D import (b2World, b2PolygonShape, b2_dynamicBody, b2DistanceJointDef, b2WeldJointDef)
import random
import numpy as np
env = gym.make('CartPole-v0')

import torch
from Box2D import b2FixtureDef
import random
import torch.nn as nn
import torch.optim as optim

class TowerBuildingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, pixel_x, pixel_y, goal_width, goal_height, grid_size, max_joints):
        # 1. Pygame Initialization
        pygame.init()

        # 2. Box2D World Initialization
        self.world = b2World(gravity=(0, -10), doSleep=True) # Box2D world
        self.screen = pygame.display.set_mode((pixel_x, pixel_y))
        # ... Load game assets, etc...
        self.goal_width = goal_width
        self.goal_height = goal_height
        self.max_joints = max_joints
        self.grid_size = grid_size
        self.cell_size = (pixel_x // grid_size, pixel_y // grid_size)
        self.clock = pygame.time.Clock()

        self.block_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

        self.ppm = 20.0  # pixels per meter
        #self.world.Step(1.0 / 60, 6, 2) # 60Hz, 6 velocity iterations, 2 position iterations

        self.ground = self.world.CreateStaticBody(position=(0, 0))
        self.ground.CreateEdgeFixture(vertices=[(-1500, 0), (1500, 0)], density=1, friction=0.3)

        self.tower_grid = np.zeros((goal_width, goal_height))


        # ... Create Box2D bodies, fixtures, joints, etc...

        self.blocks = []
        # 3. Define action and observation spaces
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(4,))
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2, 2])

        # Place the first block
        new_body = self.world.CreateDynamicBody(
            position=(300/self.ppm, 2/self.ppm),
            angle=random.choice([0, 45, 90]) * (np.pi / 180),
        )
        new_block = new_body.CreatePolygonFixture(box=(2 * self.cell_size[0]/2/self.ppm, self.cell_size[1]/2/self.ppm), density=1, friction=0.3)
        self.blocks.append(new_block)
        
    def step(self, action):
        # 4. Execute action (e.g. apply forces to Box2D bodies, etc...)
        # 5. Update Box2D world, get observations, etc...

        valid_placement_found = False
        valid_cells = []  # Store the coordinates of valid cells

        for grid_x in range(self.grid_size):
            for grid_y in range(self.grid_size):
                if self.is_valid_placement(grid_x, grid_y, max_distance=2):
                    valid_cells.append((grid_x, grid_y))

        if valid_cells:
            grid_x, grid_y = random.choice(valid_cells)
            valid_placement_found = True
        if not valid_placement_found:
            pass
        
        # block_x and block_y come from action
        block_x_coord = grid_x * self.cell_size[0] + self.cell_size[0] // 2
        block_y_coord = grid_y * self.cell_size[1] + self.cell_size[1] // 2

        new_body = self.world.CreateDynamicBody(
            position=(block_x_coord/self.ppm, block_y_coord/self.ppm),
            angle=random.choice([0, 45, 90]) * (np.pi / 180),
        )
        new_block = new_body.CreatePolygonFixture(box=(2 * self.cell_size[0]/2/self.ppm, self.cell_size[1]/2/self.ppm), density=1, friction=0.3)
        self.blocks.append(new_block)
        # self.tower_grid[grid_y, grid_x] # Mark the grid cell as occupied

        # 6. Observation:
        new_observation = self.get_observation()
        reward = self.calcualte_reward() # Calculate reward based on new state and goals
        done = self.check_done() # Check if episode is done (e.g. if agent fell off the screen, etc...)
        info = [] # Additional information (e.g. for debugging)
        return new_observation, reward, done

    def is_valid_placement(self, grid_x, grid_y, max_distance = 2):
        grid_x_coord, grid_y_coord = self.grid_to_world_coords(grid_x, grid_y)
        for existing_block in self.blocks:
            block_x_coord, block_y_coord = existing_block.body.position * self.ppm
            if abs(block_x_coord - grid_x_coord) < max_distance * self.cell_size[0] and abs(block_y_coord - grid_y_coord) < max_distance * self.cell_size[1]:
                return True
        return False
    
    def get_random_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r, g, b)
    
    def grid_to_world_coords(self, grid_x, grid_y):
        return grid_x * self.cell_size[0] + self.cell_size[0] // 2, grid_y * self.cell_size[1] + self.cell_size[1] // 2

    def get_observation(self):
        # Build observation vector: block positions, tower height/width, joint info...
        # ... get current tower state ...
        # ... get target tower shape ...
        # ... get joint usage ifo ...
        # ... get block count ...
        max_height_coord = 0
        min_x_coord, max_x_coord = float('inf'), -float('inf') # Initialize for finding width boundaries
        for block in self.blocks:
            leftest_vertex = float('inf')
            rightest_vertex = -float('inf')
            highest_vertex = -float('inf')

            for block in self.blocks:
                for vertex in block.shape.vertices:
                    x_coord, y_coord = block.body.transform * vertex * self.ppm
                    leftest_vertex = min(leftest_vertex, x_coord)
                    rightest_vertex = max(rightest_vertex, x_coord)
                    highest_vertex = max(highest_vertex, y_coord)

            highest_vertex = highest_vertex  # Convert y-coordinate to screen coordinates

            '''
            leftest_vertex /= self.ppm  # Convert x-coordinate to world coordinates
            rightest_vertex /= self.ppm  # Convert x-coordinate to world coordinates

            leftest_vertex = int(leftest_vertex)
            rightest_vertex = int(rightest_vertex)
            highest_vertex = int(highest_vertex)

            grid_x_coord, grid_y_coord = block.body.position
            max_height_coord = max(max_height_coord, grid_y_coord)
            min_x_coord = min(min_x_coord, grid_x_coord)
            max_x_coord = max(max_x_coord, grid_x_coord)
            '''
            max_height_coord = max(max_height_coord, highest_vertex)
            min_x_coord = min(min_x_coord, leftest_vertex)
            max_x_coord = max(max_x_coord, rightest_vertex)
        
        width = (max_x_coord - min_x_coord)
        height = max_height_coord
        num_blocks = len(self.blocks)

        return np.array([width, height, num_blocks])
    
    def calcualte_reward(self):
        # ... Calculate reward based on current state and goals
        progress_reward = self.calculate_progress()
        stability_reward = -0.01 * self.calculate_stability()[0] - 0.01 * self.calculate_stability()[1]
        win_bonus = 2 if self.check_win() else 0
        block_efficiency = self.calculate_efficiency()
        return progress_reward + stability_reward + win_bonus + block_efficiency
    
    def calculate_progress(self):
        [width, height] = self.get_observation()[:2]
        current_volume = width * height
        target_volume = self.goal_width * self.goal_height
        volume_difference = abs(target_volume - current_volume)
        progress_reward = 1.0 / (1 + volume_difference)
        return progress_reward
    
    def calculate_stability(self):
        average_speed = 0
        max_speed = 0
        
        for block in self.blocks:
            speed = block.body.linearVelocity.length
            average_speed += speed
            max_speed = max(max_speed, speed)

        average_speed /= len(self.blocks)

        return [average_speed, max_speed]

    def calculate_efficiency(self):
        num_blocks = self.get_observation()[2]
        return -0.01 * num_blocks
    
    def check_done(self):
        # Did the tower reach the goal? Did it collapse? etc...
        if self.check_win():
            return True
        else:
            return False
    
    def check_win(self):
        # Did the tower reach the goal?
        # ... Compare 'self.tower_grid' (or your representation) to 'self.target_tower'...
        win = False
        width, height = self.get_observation()[:2]
        #if width >= self.goal_width and height >= self.goal_height:
        if height >= self.goal_height:
            win = True
        return win
    
    def reset(self):
        # ... Reset Box2D world, get initial observations, etc...
        return self.get_observation()
    
    def render(self):
        # Clear the screen (Example: Fill with white)
        self.screen.fill((255, 255, 255))

        # Example: Draw the ground plane
        ground_y = self.ground.position.y  # Assuming the ground is at y=0
        #pygame.draw.rect(self.screen, (100, 100, 100), (0, ground_y, self.pixel_x, self.pixel_y - ground_y))

        # Draw blocks 
        for i, block in enumerate(self.blocks):
            self.draw_block(block.body, self.block_colors[i % len(self.block_colors)])

    def draw_block(self, body, color):
        for fixture in body.fixtures:
            shape = fixture.shape
            vertices = [(body.transform * v) * self.ppm for v in shape.vertices]
            vertices = [(v[0], 600 - v[1]) for v in vertices]
            pygame.draw.polygon(self.screen, color, vertices)


    def close(self):
        pygame.quit()

