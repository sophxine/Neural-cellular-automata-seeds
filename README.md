# Neural cellular automata seeds

This convolutional neural network tries to approximate a two state cellular automata rule(any kind of rule works), and create variations of them with a threshold. For example it can learn the rules of Conway's Game of Life near perfectly.

It can be used for exploring rules, because the neural network learns rules the patterns exist in and can create many different seeds. The seeds can be tweaked with the threshold to have the rule be more like how you want it.
It is not a neural cellular automata, it places cells on the grid and for this example it learns that it needs to have a 3*3 neighborhood.
I call the model weights "seeds", and the specific combination of seeds and threshold "rules".

The rule it approximates in this example is B2/S23, it can learn to near perfectly simulate it in around 15 training phases with the default soup size and generations.

# Controls
Left mouse button: draw
Right mouse button: erase
Space: pause/unpause
S: save model with the name model_weights
Tab: simulate a new soups
