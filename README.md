# cuda-neat
Parallel evaluation of NEAT genomes.

Specifically, enables the execution of a single NEAT neural network on many data in parallel. This is useful when you need to test each genome on a large data set. There will of course be no increase in performance (probably a decrease, due to the GPU bandwidth problem) on a small dataset.

The kernel is set up to execute a network with 1 hidden layer (3 executions). It can be easily altered for fewer. As well, it is set up for a one dimensional output, which can also be altered. Both of these parameters will be made more easily customizable when I get around to it.

Requires:

CUDA toolkit, MultiNEAT, PyCUDA

The setup is a little unusual. Before running setup.py for MultiNEAT, you should place the provided NeuralNetwork.cpp file into MultiNEAT/lib, replacing the existing file. This alters the code so that the Output() method outputs the parameters of the network instead of the actual output of the network. WARNING: This breaks the normal functionality of this function. This was the easiest solution for my use case, but of course not optimal. If you want a normally functioning version of MultiNEAT you should maintain that separately from this installation.
