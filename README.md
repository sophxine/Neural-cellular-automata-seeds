# Neural cellular automata seeds

The web version has a memory leakage issue so if you use it proceed with caution, it should stop before crashing the browser though.
Flashing can occur in all versions, except NCAv1 when it is unmodified.

This convolutional neural network is a neural cellular automata that tries to approximate a two state cellular automata rule(any kind of rule works), and create variations of them with a threshold. For example it can learn the rules of Conway's Game of Life as if it was Conway's Game of Life.

The example model cgol.h5 is Conway's Game of Life trained to have loss of approximately 0.00000000001.

It can be used for exploring rules, because the neural cellular automata learns rules the patterns exist in and can create many different seeds. The seeds can be tweaked with the threshold to have the rule be more like how you want it.
I call the model weights "seeds", and the specific combination of seeds and threshold "rules".

The rule it approximates in this example is B2/S23, it can learn to near perfectly simulate it in around 15 training phases with the default soup size, neural network architecture and generations.

Loading only works as intended for models in NCAv1.py.

The web version has a few differences from the python versions.
Currently drawing and erasing doesnt work on the website version, and you cant save it.


# Controls 
1. Left mouse button: draw
2. Right mouse button: erase
3. Space: pause/unpause
4. s: saves the model with the name model_weights, or saves the generator and discriminator model weights
5. Tab: simulate a new soups
6. c: clear grid
7. 1: increase threshold - less black cells (I'm not sure whether it's more intuitive to swap 1 and q or not)
8. q: decrease threshold - more white cells
9. 2: increase threshold change rate
11. w: decrease threshold change rate

# Versions
1. NCAv1: A simple convolutional neural network that learns the rule B2/S23, it is currently the only version with training.
2. Multi-state-NCAv1: The same but with multiple states, controlled by the same neural network.
3. NCAv2: An agent-based version of the neural cellular automata, agents that can create or kill cells. It seems to create more complex patterns with random weights.
4. ganca: A GAN(generative adverserial network) version of the neural cellular automata. It is very unstable but if you stabelize it, it produces similar results to NCAv1.

# Practical applications
Left open-ended (because I know far too little to know of any)
