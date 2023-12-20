# Currently you can't use the slider
import numpy as np
import tensorflow as tf
import pygame
from tensorflow.keras import layers


noise_dim=1
# Define the generator
generator = tf.keras.Sequential([
    layers.Input(shape=(noise_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid'),
    layers.Reshape((28, 28, 1))
])

# Define the discriminator (critic)
discriminator = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

# Clip discriminator weights
discriminator_clip_value = 0.01
for layer in discriminator.layers:
    if isinstance(layer, layers.Dense):
        weights = layer.trainable_variables[0]
        weights.assign(tf.clip_by_value(weights, -discriminator_clip_value, discriminator_clip_value))

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 10
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FPS = 239
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create the grid with random values
grid = np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH), p=[0.7, 0.3]).astype(np.uint8)

def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(tf.math.log(y_pred))

# Alternatively use this:
#def wasserstein_loss(y_true, y_pred):
    #return tf.reduce_mean(y_true * y_pred)

# Information Theoretic Regularization Weight
info_reg_weight = 0.9

# Kullback-Leibler Divergence
def kl_divergence(y_true, y_pred):
    # Compute KL divergence
    kl = tf.reduce_sum(y_true * tf.math.log(y_true / (y_pred + 1e-8)), axis=-1)
    return tf.reduce_mean(kl)

# Wasserstein GAN Loss with Information Theoretic Regularization
def info_theoretic_wasserstein_loss(y_true, y_pred):
    # Wasserstein Loss
    wasserstein_loss = -tf.reduce_mean(y_pred)
    
    # KL Divergence Regularization
    kl_regularization = info_reg_weight * kl_divergence(y_true, y_pred)
    
    # Total Loss
    total_loss = wasserstein_loss + kl_regularization
    
    return total_loss

# Create a simple generator and discriminator for the GAN
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(GRID_HEIGHT, GRID_WIDTH, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='swish', padding='same'),
    tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(GRID_HEIGHT, GRID_WIDTH, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='mean_squared_error')
# The combined GAN model
discriminator.trainable = False
gan_input = tf.keras.layers.Input(shape=(GRID_HEIGHT, GRID_WIDTH, 1))
generated_state = generator(gan_input)
gan_output = discriminator(generated_state)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss=wasserstein_loss)


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
    global text2
    pygame.draw.rect(screen, slider_color, (slider_x, slider_y, slider_width, slider_height))
    pygame.draw.circle(screen, slider_color, (int(slider_x + slider_width * threshold), slider_y + slider_height // 2), 9)
    if display_threshold==True:
        text = font.render(f"Threshold: {threshold:.5f}", True, (0, 128, 255))
    if display_threshold_change_rate==True:
        text2 = font.render(f"threshold change rate: {threshold_change_rate:.5f}", True, (0, 128, 255))
        
    screen.blit(text, (slider_x, slider_y - 30))
    screen.blit(text2, (slider_x, slider_y+ 30))

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
                
running = True


#Change these to finetune it
threshold = 0.5 # Initial threshold for how likely a cells are to be created, lower values is more cells. This value does not affect the training, it only affects the simulation.
threshold_change_rate=0.001 #Initial threshold change rate 
display_threshold=True
display_threshold_change_rate=True
symmetry=False #WIP

clear_grid = False
drawing = False  
drawing_paused = False
simulation_paused = False
manual_pause = False
dragging = False
if symmetry == True:
    # Define the convolution kernel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
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

                
        if event.type == pygame.KEYDOWN: #Not working
            if event.key == pygame.K_p:  
                draw_radius += 1
            elif event.key == pygame.K_m:  
                draw_radius = max(1, draw_radius - 1)
    # Preview the grid when dragging the slider
# Preview the grid when dragging the slider
    # Preview the grid when dragging the slider
    if dragging:
        screen.fill(WHITE)
        preview_grid = np.where(generated_state > threshold, 1, 0)
        for y in range(min(GRID_HEIGHT, preview_grid.shape[0])):
            for x in range(min(GRID_WIDTH, preview_grid.shape[1])):
                if np.any(preview_grid[y, x] == 1):
                    cell_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, BLACK, cell_rect, 0)  # Filled rectangle for the cell
                    pygame.draw.rect(screen, WHITE, cell_rect, 2)  # Thicker border
        grid = np.where(generated_state > threshold, 1, 0)


    if clear_grid:
        screen.fill(WHITE)
        clear_grid = False
    else:
        if not drawing_paused and not simulation_paused:
            if symmetry == True:
                # Apply convolution for rotational invariance and symmetry
                input_data = np.expand_dims(grid, axis=-1)
                convolved_grid = scipy.signal.convolve2d(grid, kernel, mode='same', boundary='wrap')
                generated_state = generator.predict(np.expand_dims(input_data, axis=0))
                grid = np.where(generated_state > threshold, 1, 0)
            
            else:
# Prepare input data for the GAN
                input_data = np.expand_dims(grid, axis=-1)

                # Train the discriminator
                real_labels = np.ones((1, 60, 80, 1))
                fake_labels = np.zeros((1, 60, 80, 1))

                discriminator.trainable = True
                real_loss = discriminator.train_on_batch(np.expand_dims(input_data, axis=0), real_labels)
                generated_state = generator.predict(np.expand_dims(input_data, axis=0))
                fake_loss = discriminator.train_on_batch(generated_state, fake_labels)

                # Train the generator
                noise = np.random.normal(0, 1, size=(1, GRID_HEIGHT, GRID_WIDTH, 1))
                discriminator.trainable = False
                gan_loss = gan.train_on_batch(noise, real_labels)
                # Update the grid with the generated state
                generated_state = generator.predict(np.expand_dims(input_data, axis=0))
                grid = np.where(generated_state > threshold, 1, 0)[0, :, :, 0]
        screen.fill(WHITE)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if grid[y, x] == 1:
                    pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    draw_slider()
    pygame.display.flip()

pygame.quit()
