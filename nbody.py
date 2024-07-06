import pygame
import numpy as np
import torch

screen_size = screen_width, screen_height = 1920, 1080
screen = pygame.display.set_mode(screen_size)

clock = pygame.time.Clock()
fps_limit = 60

#circle 
colorcircle = "gray"

NUM_PARTICLES = 4500

@torch.compile
def update(positions, velocities, dt):
    diffs = torch.unsqueeze(positions, 1) - positions
    diffs_sqr = torch.clip((diffs**2).sum(2), 7**2, None)
    force = (diffs_sqr > 7**2) * 0.00001 / diffs_sqr / torch.sqrt(diffs_sqr) * diffs.T

    vel_diffs = torch.unsqueeze(velocities, 1) - velocities
    vel_force = (diffs_sqr > 7**2) * 0.005 / diffs_sqr / torch.sqrt(diffs_sqr) * ((vel_diffs * diffs).sum(2) / diffs_sqr * diffs.T)

    velocities += (torch.sum(force.T, axis=0) + torch.sum(vel_force.T, axis=0)) * dt

    positions += velocities * dt
    return positions, velocities

positions = np.random.random(size=(NUM_PARTICLES,2)) * 300 - 150
velocities = np.random.random(size=(NUM_PARTICLES,2)) * 0.025
velocities[0] = -velocities[1]
positions[0] = -positions[1]

positions = torch.tensor(positions).to('cuda:0')
velocities = torch.tensor(velocities).to('cuda:0')

for i in range(NUM_PARTICLES):
    angle = np.random.random() * np.pi * 2
    dist = np.random.random() * 0.8 + 0.001
    positions[i, 0] = np.cos(angle) * 200 * dist
    positions[i, 1] = np.sin(angle) * 200 * dist
    velocities[i, 1] = np.cos(angle) * 0.012
    velocities[i, 0] = -np.sin(angle) * 0.012

while True:
    clock.tick(fps_limit)
    time = pygame.time.get_ticks()
    dt = max(clock.get_time(), 1/10)
    dt = 100

    positions, velocities = update(positions, velocities, dt)

    # fill the screen with black (otherwise, the circle will leave a trail)
    screen.fill("black")
    # redraw the circle
    pos = positions.cpu().numpy()
    com = np.mean(pos, 0)
    for i in range(NUM_PARTICLES):
        posx = (pos[i, 0] - com[0])
        posy = (pos[i, 1] - com[1])
        posx += 1920 // 2
        posy += 1080 // 2
        pygame.draw.circle(screen, colorcircle, (posx, posy), 1)

    pygame.display.flip()

pygame.quit()