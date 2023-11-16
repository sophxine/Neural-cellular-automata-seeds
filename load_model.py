#To use this script enter the filename of your model in loaded_model.

import numpy as np
import tensorflow as tf
import pygame

threshold = 0.909


# Constants
WIDTH, HEIGHT = 800, 600
CELL_SIZE =3
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FPS = 220

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize pygame
pygame.init()

# Create the grid with random values
grid = np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH))

# Create the pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Loaded Seed")

clock = pygame.time.Clock()

# Load the saved model
loaded_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(GRID_HEIGHT, GRID_WIDTH, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    #tf.keras.layers.Conv2D(128, (3, 3), activation='swish', padding='same'),
    #tf.keras.layers.Conv2D(128, (3, 3), activation='sigmoid', padding='same'
    tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])
loaded_model.compile(optimizer='adam', loss='binary_crossentropy')
loaded_model.load_weights("diverse_rakes_and_puffers.h5")

# Function to draw on the grid
def draw_on_grid(mouse_pos, value=1):
    x, y = mouse_pos
    x //= CELL_SIZE
    y //= CELL_SIZE

    # Ensure the indices are within bounds
    x = np.clip(x, 0, GRID_WIDTH - 1)
    y = np.clip(y, 0, GRID_HEIGHT - 1)

    grid[y, x] = value  # Set the cell at the clicked position
# Main loop
running = True
clear_grid = False
drawing = False  # Variable to track drawing state

drawing_paused = False
simulation_paused = False
manual_pause = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
                draw_on_grid(pygame.mouse.get_pos())
                drawing_paused = True
                simulation_paused = True
            elif event.button == 3:  # Right mouse button
                drawing = True
                draw_on_grid(pygame.mouse.get_pos(), value=0)  # Erase with right button
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
    if clear_grid:
        screen.fill(WHITE)
        clear_grid = False
    else:
        if not drawing_paused and not simulation_paused:
            predicted_state = loaded_model.predict(np.expand_dims(grid, axis=(0, -1)))[0, :, :, 0]
            grid = np.where(predicted_state > threshold, 1, 0)
        screen.fill(WHITE)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if grid[y, x] == 1:
                    pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()

