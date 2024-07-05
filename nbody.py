import pygame
import numpy as np
import torch

screen_size = screen_width, screen_height = 1920, 1080
screen = pygame.display.set_mode(screen_size)

clock = pygame.time.Clock()
fps_limit = 60

#circle 
colorcircle = "gray"

NUM_PARTICLES = 3000

positions = np.random.random(size=(NUM_PARTICLES,2)) * 300 - 150
velocities = np.random.random(size=(NUM_PARTICLES,2)) * 0.02
velocities[0] = -velocities[1] - velocities[2]
positions[0] = -positions[1] - positions[2]

positions = torch.tensor(positions).to('cuda:1')
velocities = torch.tensor(velocities).to('cuda:1')

for i in range(NUM_PARTICLES):
    angle = np.random.random() * np.pi * 2
    dist = np.random.random() * 0.5 + 1
    positions[i, 0] = np.cos(angle) * 200 * dist
    positions[i, 1] = np.sin(angle) * 200 * dist
    velocities[i, 1] = np.cos(angle) * 0.0523
    velocities[i, 0] = -np.sin(angle) * 0.0523

while True:
    clock.tick(fps_limit)
    time = pygame.time.get_ticks()
    dt = clock.get_time()

    diffs = torch.unsqueeze(positions, 1) - positions
    diffs_sqr = diffs[:, :, 0]**2 + diffs[:, :, 1]**2 + 0.0000001
    diffs_sqr = torch.clip(diffs_sqr, 7**2, None)
    force = 0.001 / diffs_sqr / torch.sqrt(diffs_sqr) * diffs.T
    force = torch.sum(force.T, axis=0)

    vel_diffs = torch.unsqueeze(velocities, 1) - velocities
    vel_force = 0.3 / diffs_sqr**2 / torch.sqrt(diffs_sqr) * vel_diffs.T
    vel_force *= (diffs_sqr > 12**2) * (diffs_sqr < 40**2)
    vel_force = torch.sum(vel_force.T, axis=0)
    force += vel_force

    velocities += force * dt

    positions += velocities * dt

    # fill the screen with black (otherwise, the circle will leave a trail)
    screen.fill("black")
    # redraw the circle
    pos = positions.cpu().numpy()
    for i in range(NUM_PARTICLES):
        posx = pos[i, 0]
        posy = pos[i, 1]
        posx += 1920 // 2
        posy += 1080 // 2
        pygame.draw.circle(screen, colorcircle, (posx, posy), 1)

    pygame.display.flip()

pygame.quit()