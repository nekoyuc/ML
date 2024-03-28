import pygame
from Box2D import (b2World, b2PolygonShape, b2_dynamicBody, b2DistanceJointDef, b2WeldJointDef)
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

blocks = [create_block(random.randint(0, width-50), 600, 50, 50) for _ in range(20)]
# Add a second row of blocks starting higher
blocks.extend([create_block(random.randint(0, width-50), 800, 50, 50) for _ in range(20)])

# Add a floor
ground = world.CreateStaticBody(position=(0, 0))
ground.CreateEdgeFixture(vertices=[(0, 0), (width/ppm, 0)], density=1, friction=0.3)

distance_joints = []
for i in range(3):
    block1 = random.choice(blocks)
    block2 = random.choice([block for block in blocks if block != block1])  # Ensure block2 is different from block1
    distance_joint = b2DistanceJointDef()
    distance_joint.bodyA = block1
    distance_joint.bodyB = block2
    distance_joint.localAnchorA = (0, 0)
    distance_joint.localAnchorB = (0, 0)
    distance_joint.length = 4.0
    world.CreateJoint(distance_joint)
    distance_joints.append(distance_joint)

weld_joints = []
for i in range(3):
    block1 = random.choice(blocks)
    block2 = random.choice([block for block in blocks if block != block1])  # Ensure block2 is different from block1
    weld_joint = b2WeldJointDef()
    weld_joint.bodyA = block1
    weld_joint.bodyB = block2
    weld_joint.localAnchorA = (block1.fixtures[0].shape.vertices[0][0], block1.fixtures[0].shape.vertices[0][1])
    weld_joint.localAnchorB = (block2.fixtures[0].shape.vertices[1][0], block2.fixtures[0].shape.vertices[1][1])
    world.CreateJoint(weld_joint)
    weld_joints.append(weld_joint)

def draw_block(body, color):
    for fixture in body.fixtures:
        shape = fixture.shape
        vertices = [(body.transform * v) * ppm for v in shape.vertices]
        vertices = [(v[0], height - v[1]) for v in vertices]
        pygame.draw.polygon(screen, color, vertices)

def draw_distance_joint(distance_joint):
    bodyA = distance_joint.bodyA
    bodyB = distance_joint.bodyB
    posA = bodyA.transform * distance_joint.localAnchorA
    posB = bodyB.transform * distance_joint.localAnchorB
    posA = (posA[0] * ppm, height - posA[1] * ppm)
    posB = (posB[0] * ppm, height - posB[1] * ppm)
    pygame.draw.line(screen, (255, 0, 0), posA, posB, 4)

def draw_weld_joint(weld_joint):
    bodyA = weld_joint.bodyA
    bodyB = weld_joint.bodyB
    posA = bodyA.transform * weld_joint.localAnchorA
    posB = bodyB.transform * weld_joint.localAnchorB
    posA = (posA[0] * ppm, height - posA[1] * ppm)
    posB = (posB[0] * ppm, height - posB[1] * ppm)
    pygame.draw.circle(screen, (0, 0, 255), posA, 6)
    pygame.draw.circle(screen, (0, 0, 255), posB, 6)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # Update Box2D world
    # Note that the second and third arguments are the velocity and position iterations, respectively
    world.Step(1.0/60, 10, 10)

    for i, block in enumerate(blocks):
        draw_block(block, BLOCK_COLORS[i % len(BLOCK_COLORS)])

    for distance_joint in distance_joints:
        draw_distance_joint(distance_joint)

    for weld_joint in weld_joints:
        draw_weld_joint(weld_joint)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
