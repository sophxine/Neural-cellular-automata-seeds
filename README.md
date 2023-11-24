# Neural cellular automata seeds

This convolutional neural network tries to approximate a two state cellular automata rule(any kind of rule works), and create variations of them with a threshold. For example it can learn the rules of Conway's Game of Life as if it was Conway's Game of Life.

The example model cgol.h5 is Conway's Game of Life trained to 99.999999999% accuracy

It can be used for exploring rules, because the neural network learns rules the patterns exist in and can create many different seeds. The seeds can be tweaked with the threshold to have the rule be more like how you want it.
It is not a neural cellular automata, it sets the grid cell states for generation and for this example it learns that it needs to have a 3*3 neighborhood.
I call the model weights "seeds", and the specific combination of seeds and threshold "rules".

The rule it approximates in this example is B2/S23, it can learn to near perfectly simulate it in around 15 training phases with the default soup size, neural network architecture and generations.


# Controls
1. Left mouse button: draw
2. Right mouse button: erase
3. Space: pause/unpause
4. S: save model with the name model_weights
5. Tab: simulate a new soups
