#Neural Cellular Automata using agents that can create or kill cells, it seems to create more complex patterns.

import numpy as np
import tensorflow as tf
import pygame

# Initialize pygame
pygame.init()
threshold_change_rate=0.001 #Initial threshold change rate 

# Constants
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 10
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FPS = 60
threshold = 0.33

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create the grid with random values
grid = np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH), p=[0.7, 0.3]).astype(np.uint8)

# Create the pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent-based neural network")

clock = pygame.time.Clock()

# Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(GRID_HEIGHT, GRID_WIDTH, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(16, (3, 3), activation='swish', padding='same'),
    tf.keras.layers.Conv2D(3, (3, 3), activation='softmax', padding='same')  # Output for move, place, kill
])

# Function to draw on the grid
def draw_on_grid(mouse_pos, value=1):
    x, y = mouse_pos
    x //= CELL_SIZE
    y //= CELL_SIZE

    # Ensure the indices are within bounds
    x = np.clip(x, 0, GRID_WIDTH - 1)
    y = np.clip(y, 0, GRID_HEIGHT - 1)

    grid[y, x] = value  # Set the cell at the clicked position

draw_radius = 0

# Function to draw on the grid with a specified radius(not working)
def draw_on_grid_with_radius(mouse_pos, value=1, radius=1):
    x, y = mouse_pos
    x //= CELL_SIZE
    y //= CELL_SIZE

    # Draw a square with the specified radius
    for i in range(max(0, x - radius), min(GRID_WIDTH, x + radius + 1)):
        for j in range(max(0, y - radius), min(GRID_HEIGHT, y + radius + 1)):
            # Ensure the indices are within bounds
            if 0 <= i < GRID_WIDTH and 0 <= j < GRID_HEIGHT:
                grid[j, i] = value

# Set up slider parameters
slider_x = 20
slider_y = 40
slider_width = 100
slider_height = 10
slider_color = (0, 128, 255)

# Set up font
font = pygame.font.Font(None, 32)

# Function to draw the float slider
def draw_slider():
    pygame.draw.rect(screen, slider_color, (slider_x, slider_y, slider_width, slider_height))
    pygame.draw.circle(screen, slider_color, (int(slider_x + slider_width * threshold), slider_y + slider_height // 2), 15)
    text = font.render(f"Threshold: {threshold:.2f}", True, (0, 128, 255))
    screen.blit(text, (slider_x, slider_y - 30))

# Compile the model
running = True
clear_grid = False
drawing = False
drawing_paused = False
simulation_paused = False
manual_pause = False
dragging = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if slider_x <= event.pos[0] <= slider_x + slider_width and slider_y <= event.pos[1] <= slider_y + slider_height:
                dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            dragging = False
        if event.type == pygame.MOUSEMOTION and dragging:
            threshold = (event.pos[0] - slider_x) / slider_width
            threshold = np.clip(threshold, 0.0, 1.0)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
                draw_on_grid_with_radius(pygame.mouse.get_pos(),value=1, radius=draw_radius)
                drawing_paused = True
                simulation_paused = True
            elif event.button == 3:  # Right mouse button
                drawing = True
                draw_on_grid_with_radius(pygame.mouse.get_pos(),value=0,radius=draw_radius)  # Erase with right button
                drawing_paused = True
                simulation_paused = True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 or event.button == 3:  # Left or right mouse button
                drawing = False
                drawing_paused = False
                if not manual_pause:  # Only resume if not manually paused
                    simulation_paused = False
        if event.type == pygame.MOUSEMOTION and drawing:
            if event.buttons[0]:  # Left button is pressed
                draw_on_grid(pygame.mouse.get_pos())
            elif event.buttons[2]:  # Right button is pressed
                draw_on_grid(pygame.mouse.get_pos(), value=0)  # Erase with right button
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                simulation_paused = not simulation_paused
                manual_pause = simulation_paused  # Update manual pause state
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
                clear_grid = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                grid = np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH)).astype(np.uint8)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                model.save_weights(f"model_weights.h5")
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                threshold+=threshold_change_rate
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                threshold-=threshold_change_rate
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                threshold_change_rate=threshold_change_rate/10
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_2:
                threshold_change_rate=threshold_change_rate*10 

    if clear_grid:
        screen.fill(WHITE)
        clear_grid = False
    # Preview the grid when dragging the slider
    if dragging:
        screen.fill(WHITE)
        preview_grid = np.where(action_probabilities[0, :, :, 1] > threshold, 1, 0)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if preview_grid[y, x] == 1:
                    cell_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, BLACK, cell_rect, 0)  # Filled rectangle for the cell
                    pygame.draw.rect(screen, WHITE, cell_rect, 2)  # Thicker border
        grid = np.where(action_probabilities[0, :, :, 1] > threshold, 1, 0)


    else:
        if not drawing_paused and not simulation_paused:
            # Predict the action probabilities for each cell
            action_probabilities = model.predict(np.expand_dims(grid, axis=(0, -1)))

            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    # Sample the action based on the predicted probabilities
                    action = np.random.choice([0, 1, 2], p=action_probabilities[0, y, x])

                    # Take action based on the sampled value
                    if action == 0:  # Move
                        pass  # Implement move logic
                    elif action == 1:  # Place
                        grid[y, x] = 1
                    elif action == 2:  # Kill
                        grid[y, x] = 0

        # Apply threshold to the grid based on the neural network's prediction
        if not drawing_paused and not simulation_paused:
            grid = np.where(action_probabilities[0, :, :, 1] > threshold, 1, 0)
        
                        
        screen.fill(WHITE)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if grid[y, x] == 1:
                    pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    draw_slider()
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
