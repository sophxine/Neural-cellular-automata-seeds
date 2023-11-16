
# The rule it approximates in this example is B2/S23

# Tips:
# If it's a lot too explosive or sparse consider modifying the threshold for exploring, though you can also explore explosive/sparse seeds changing the threshold. You can be more exact than the slider by modifying the threshold in the code.InteractiveInterpreter
# If you find a seed you find interesting don't forget to press 's' to save!


# Todo:
# Start training with a new random soup and initial rule using r
# Start training of current iteration with e(like mutating)
# It should ask for threshold, generations to simulate and how long to train

# How the save and load training iteration should work:
# Save: saves to the folder under a name you pick in a popup window(more elegant after)
# Saved model should be named the threshold it is at because it might depend on it.
# Load: loads a save from the folder under a name you pick, if there is none it asks you to retype. It also asks for you to set a threshold.

# Option to display training in matplot
# Option to start training with a saved model at start
# Options to force symmetry

import numpy as np
import tensorflow as tf
import pygame

#Change these to finetune it
threshold = 0.65 # Initial threshold for how likely a cells are to be created, lower values is more cells. This value does not affect the training, it only affects the simulation.
generations = 11 # Generati0466ons to simulate
trainingphases=9 # How long to train


# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 4
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FPS =60
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create a convolutional neural network for cell simulation, you can change the model architechture as you want.
# Swish also is suitable as an activition function
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(GRID_HEIGHT, GRID_WIDTH, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    #tf.keras.layers.Conv2D(128, (3, 3), activation='swish', padding='same'),
    #tf.keras.layers.Conv2D(128, (3, 3), activation='sigmoid', padding='same'),
    tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

# Compile the model
# For the optimizer adam and nadam are suitable
# For the loss function BCE and MeanSquaredError are suitable
model.compile(optimizer='adam', loss='binary_crossentropy')



# Create the grid with random values
grid = np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH)).astype(np.uint8)

# Create the pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Cellular Automata Seeds")
clock = pygame.time.Clock()

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

# Simulate a cellular automata soup for specified amount of generations
initial_states = [grid.copy()]
for _ in range(generations):
    next_grid = np.zeros_like(grid)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            total_neighbors = np.sum(grid[max(0, y - 1):min(GRID_HEIGHT, y + 2), max(0, x - 1):min(GRID_WIDTH, x + 2)]) - grid[y, x]
            if grid[y, x] == 1:
                if total_neighbors in (2, 3):
                    next_grid[y, x] = 1
            else:
                if total_neighbors ==2: #Set this to 3 for Conway's Game of Life
                    next_grid[y, x] = 1            
    grid = next_grid.copy()
    initial_states.append(grid.copy())

# Reshape the states for training
training_data = np.array(initial_states[:-1])
target_data = np.array(initial_states[1:])


#To predict for several generations(i.e 3) in the future set it like this:

#training_data = np.array(initial_states[:-3])  
#target_data = np.array(initial_states[3:])

# Start training based on the generated generations
model.fit(training_data[..., np.newaxis], target_data[..., np.newaxis], epochs=trainingphases, batch_size=1, verbose=1)
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

        if event.type == pygame.KEYDOWN: #Not working
            if event.key == pygame.K_p:  
                draw_radius += 1
            elif event.key == pygame.K_m:  
                draw_radius = max(1, draw_radius - 1)        
    if clear_grid:
        screen.fill(WHITE)
        clear_grid = False
    else:
        if not drawing_paused and not simulation_paused:
            predicted_state = model.predict(np.expand_dims(grid, axis=(0, -1)))[0, :, :, 0]
            grid = np.where(predicted_state > threshold, 1, 0)

        screen.fill(WHITE)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if grid[y, x] == 1:
                    pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    draw_slider()
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()

