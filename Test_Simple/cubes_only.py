import pygame
from Box2D import (b2World, b2PolygonShape, b2_dynamicBody)
import random

# Pygame setup
pygame.init()
width, height = 800, 600
ppm = 20.0  # pixels per meter
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLOCK_COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

# Box2D world setup
world = b2World(gravity=(0, -10), doSleep=True)

def create_block(x, y, width, height):
    body = world.CreateDynamicBody(position=(x/ppm, y/ppm))
    box = body.CreatePolygonFixture(box=(width/2/ppm, height/2/ppm), density=1, friction=0.3)
    return body

blocks = [create_block(random.randint(0, width-50), 600, 50, 50) for _ in range(10)]
# Add a second row of blocks starting higher
blocks.extend([create_block(random.randint(0, width-50), 800, 50, 50) for _ in range(10)])
# Add a floor
ground = world.CreateStaticBody(position=(0, 0))
ground.CreateEdgeFixture(vertices=[(0, 0), (width/ppm, 0)], density=1, friction=0.3)


def draw_block(body, color):
    for fixture in body.fixtures:
        shape = fixture.shape
        vertices = [(body.transform * v) * ppm for v in shape.vertices]
        vertices = [(v[0], height - v[1]) for v in vertices]
        pygame.draw.polygon(screen, color, vertices)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # Update Box2D world
    world.Step(1.0/60, 10, 10)

    for i, block in enumerate(blocks):
        draw_block(block, BLOCK_COLORS[i % len(BLOCK_COLORS)])

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
