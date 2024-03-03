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
        # 1. Pygame Initialization
        pygame.init()

        # 2. Box2D World Initialization
        self.world = b2World(gravity=(0, -10), doSleep=True) # Box2D world
        self.screen = pygame.display.set_mode((screen_x, screen_y))
        # ... Load game assets, etc...
        self.goal_width = goal_width
        self.goal_height = goal_height
        self.max_joints = max_joints
        self.grid_size = grid_size
        self.cell_size = (screen_x // grid_size, screen_y // grid_size) # (20, 20) pixels per grid cell
        self.clock = pygame.time.Clock()

        self.block_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

        self.ppm = 20.0  # pixels per meter
        #self.world.Step(1.0 / 60, 6, 2) # 60Hz, 6 velocity iterations, 2 position iterations

        self.ground = self.world.CreateStaticBody(position=(0, 0))
        self.ground.CreateEdgeFixture(vertices=[(-1500, 0), (1500, 0)], density=1, friction=0.3)

        self.tower_grid = np.zeros((goal_width, goal_height))

        self.image_index = 0

        self.block_radius = self.cell_size[0] * 0.5 * np.sqrt(5) # 22.3 pixels
        
        self.max_height_coord = 0
        self.min_x_coord = screen_x/2
        self.max_x_coord = screen_x/2

        self.width = 0
        self.height = 0

        # ... Create Box2D bodies, fixtures, joints, etc...

        self.blocks = []
        # Add a list to record score at each step
        self.current_score = 0
        self.steps = 0

        # (step, score, width, height)
        self.records = []
        # 3. Define action and observation spaces
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(4,))
        self.action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (2,))

        # Score function parameters
        # width reward = -alpha * (width - goal_width)
        # height reward = -beta * height ^ theta
        # stability punishment = gamma * (average_speed + max_speed)
        # efficiency punishment = delta * block_num
        self.alpha = 0.005
        self.beta = 0.0001
        self.theta = 2
        self.gamma = -0.01
        self.delta = -0.005
        
        
        # Place the first block
        new_body = self.world.CreateDynamicBody(
            position=(screen_x/self.ppm/2, 2/self.ppm),
            #angle=random.choice([0, 45, 90]) * (np.pi / 180),
            angle=random.choice([0, 90]) * (np.pi / 180),
        )
        new_block = new_body.CreatePolygonFixture(box=(2 * self.cell_size[0]/2/self.ppm, self.cell_size[1]/2/self.ppm), density=1, friction=0.3)
        self.blocks.append(new_block)

        self.new_block = new_block        
        
    def step(self, action):
        # 4. Execute action (e.g. apply forces to Box2D bodies, etc...)
        # 5. Update Box2D world, get observations, etc...

        valid_placement_found = False
        valid_cells = []  # Store the coordinates of valid cells

        for grid_x in range(self.grid_size):
            for grid_y in range(1, self.grid_size):
                if self.is_valid_placement(grid_x, grid_y, max_distance_squared=5000):
                    valid_cells.append((grid_x, grid_y))

        if valid_cells:
            grid_x, grid_y = random.choice(valid_cells)
            valid_placement_found = True
        if not valid_placement_found:
            print("No valid placement found")
            pass
        
        # block_x and block_y come from action
        block_x_coord, block_y_coord = self.grid_to_world_coords(grid_x, grid_y)

        new_body = self.world.CreateDynamicBody(
            position=(block_x_coord/self.ppm, block_y_coord/self.ppm),
            angle=random.choice([0, 45, 90]) * (np.pi / 180),
        )
        new_block = new_body.CreatePolygonFixture(box=(2 * self.cell_size[0]/2/self.ppm, self.cell_size[1]/2/self.ppm), density=1, friction=0.3)
        self.blocks.append(new_block)
        self.new_block = new_block

    def is_valid_placement(self, grid_x, grid_y, max_distance_squared):
        grid_x_coord, grid_y_coord = self.grid_to_world_coords(grid_x, grid_y)
        # Create a temporary fixture representing the potential new block

        potential_block_body = self.world.CreateDynamicBody(
            position=(grid_x_coord/self.ppm, grid_y_coord/self.ppm)
        )
        
        # Create a temporary circular fixture attached to the potential block body
        p_f = potential_block_body.CreateCircleFixture(radius=50/self.ppm, density=1, friction=0.3)
        # Render the potential block

        closest = 360000.0
        for e in self.blocks:
            if self.fixtures_overlap(p_f, e):
                self.world.DestroyBody(potential_block_body)
                return False
            else:
                block_x_coord, block_y_coord = self.grid_to_world_coords(e.body.position.x, e.body.position.y)
                distance_squared = (block_x_coord - grid_x_coord) ** 2 + (block_y_coord - grid_y_coord) ** 2
                closest =  min(distance_squared, closest)
        
        if closest < max_distance_squared:
            self.world.DestroyBody(potential_block_body)
            return True
        else:
            self.world.DestroyBody(potential_block_body)
            return False

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
 
    def update_score(self):
        # ... Calculate reward based on current state and goals
        progress = self.calculate_progress()
        stability_punishment = self.calculate_stability()[2]
        #win_bonus = 2 if self.check_win() else 0
        efficiency_punishment = self.calculate_efficiency()
        
        self.current_score = progress + stability_punishment + efficiency_punishment

    def update_records(self):
        self.steps += 1

        leftest_vertex = float('inf')
        rightest_vertex = -float('inf')
        highest_vertex = -float('inf')
        for vertext in self.new_block.shape.vertices:
            new_x_coord, new_y_coord = self.new_block.body.transform * vertext * self.ppm
            leftest_vertex = min(leftest_vertex, new_x_coord)
            rightest_vertex = max(rightest_vertex, new_x_coord)
            highest_vertex = max(highest_vertex, new_y_coord)
        
        self.max_height_coord = max(self.max_height_coord, highest_vertex)
        self.min_x_coord = min(self.min_x_coord, leftest_vertex)
        self.max_x_coord = max(self.max_x_coord, rightest_vertex)

        self.width = (self.max_x_coord - self.min_x_coord)
        self.height = self.max_height_coord

        self.records.append((self.steps, self.current_score, self.width, self.height))

    def calculate_progress(self):
        progress_x = - self.alpha * (self.width - self.goal_width)
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
        punishment = self.gamma * (average_speed + max_speed)
        return [average_speed, max_speed, punishment]
    
    def calculate_efficiency(self):
        return self.delta * len(self.blocks)
    
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

        resized_screen = gray_image.resize((84, 84), Image.BILINEAR)
        # Save gray image
        gray_image.save(f'screenshot_{self.image_index}.png')
        #resized_screen.save(f'screenshot_resized_{self.image_index}.png')

        # export the raw_screen to a file
        #pygame.image.save(raw_screen, f'screenshot_{self.image_index}.png')
        self.image_index += 1

        #return resized_screen

    def reset(self):
        self.blocks = []

        self.max_height_coord = 0
        self.min_x_coord = screen_x/2
        self.max_x_coord = screen_x/2
        self.width = 0
        self.height = 0
        self.image_index = 0

        # Place the first block
        new_body = self.world.CreateDynamicBody(
            position=(screen_x/self.ppm/2, 2/self.ppm),
            #angle=random.choice([0, 45, 90]) * (np.pi / 180),
            angle=random.choice([0, 90]) * (np.pi / 180),
        )
        new_block = new_body.CreatePolygonFixture(box=(2 * self.cell_size[0]/2/self.ppm, self.cell_size[1]/2/self.ppm), density=1, friction=0.3)
        self.blocks.append(new_block)

        self.new_block = new_block
        self.records()
    
    def render(self):
        # Clear the screen (Example: Fill with white)
        self.screen.fill((255, 255, 255))

        # Example: Draw the ground plane
        ground_y = self.ground.position.y  # Assuming the ground is at y=0
        #pygame.draw.rect(self.screen, (100, 100, 100), (0, ground_y, self.screen_x, self.screen_y - ground_y))

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

