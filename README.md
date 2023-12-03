# Neural cellular automata seeds

The web version has a memory leakage issue so if you use it proceed with caution, it should stop before crashing the browser though.
Flashing can occur in all versions, except NCAv1 when it is unmodified.

This convolutional neural network is a simple neural cellular automata that tries to approximate a two state cellular automata rule(any kind of rule works), and create variations of them with a threshold. For example it can learn the rules of Conway's Game of Life as if it was Conway's Game of Life.

The example model cgol.h5 is Conway's Game of Life trained to approximately 99.999999999% accuracy according to the loss used.

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
4. S: save model with the name model_weights
5. Tab: simulate a new soups
6. C: clear grid
7. 1: increase threshold - less black cells 
8. q: decrease threshold - more white cells
9. 2: increase threshold change rate
11. w: decrease threshold change rate
