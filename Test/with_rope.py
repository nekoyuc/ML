import pygame
from Box2D import (b2World, b2PolygonShape, b2_dynamicBody, b2RopeJoint)
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
ROPE_COLOR = (0, 0, 0)  # Black color for ropes

# Box2D world setup
world = b2World(gravity=(0, -10), doSleep=True)

def create_block(x, y, size):
    body = world.CreateDynamicBody(position=(x/ppm, y/ppm))
    vertices = [(-size/2/ppm, -size/2/ppm), (size/2/ppm, -size/2/ppm), (size/2/ppm, size/2/ppm), (-size/2/ppm, size/2/ppm)]
    box = b2PolygonShape(vertices=vertices)
    body.CreateFixture(shape=box, density=1, friction=0.3)
    return body

blocks = [create_block(random.randint(0, width-50), 600, 50) for _ in range(10)]
# Add a second row of blocks starting higher
blocks.extend([create_block(random.randint(0, width-50), 800, 50) for _ in range(10)])
# Add a floor
ground = world.CreateStaticBody(position=(0, 0))
ground.CreateEdgeFixture(vertices=[(0, 0), (width/ppm, 0)], density=1, friction=0.3)

# Randomly connect blocks with ropes
ropes = []
for i in range(len(blocks)-1):
    block1 = blocks[i]
    block2 = blocks[i+1]
    rope_joint = world.CreateRopeJoint(bodyA=block1, bodyB=block2, anchorA=(0, 0), anchorB=(0, 0), maxLength=2.0)
    ropes.append((block1, block2))

# Remove most of the ropes, leave only 3 ropes
ropes = random.sample(ropes, 3)

def draw_block(body, color):
    for fixture in body.fixtures:
        shape = fixture.shape
        vertices = [(body.transform * v) * ppm for v in shape.vertices]
        vertices = [(v[0], height - v[1]) for v in vertices]
        pygame.draw.polygon(screen, color, vertices)

def draw_rope(body1, body2, color):
    pos1 = body1.position * ppm
    pos2 = body2.position * ppm
    pygame.draw.line(screen, color, (pos1.x, height - pos1.y), (pos2.x, height - pos2.y), 2)

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

    for rope in ropes:
        draw_rope(rope[0], rope[1], ROPE_COLOR)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
